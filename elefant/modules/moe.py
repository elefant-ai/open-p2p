from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from elefant.torch import eager_assert
import math


class TokenChoiceTopKRouter(nn.Module):
    """This class implements Token Choice routing. In Token Choice top K routing, each token is
        routed to top K experts GroupedExpertssed on the router scores.

    Args:
        gate (nn.Module): Gate module to calculate the scores, typically nn.Linear(dim, num_experts).
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        experts_per_token (int): Number of experts each token will be routed to in Token Choice.
    """

    def __init__(
        self,
        *,
        gate: nn.Module,
        dim: int,
        num_experts: int,
        experts_per_token: int,
    ):
        super().__init__()
        self.gate = gate
        self.dim = dim
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token

    def forward(
        self, x: torch.Tensor, batch_size: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): shape (batch_size x seq_len, dim).

        Returns:
            routed_input (torch.Tensor):
                Tokens grouped together by experts indices with shape ``(batch_size x seq_len x top_k,)``.
            token_indices (torch.Tensor):
                Token indices for routed_input with shape ``(batch_size x seq_len x top_k,)``.
            num_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        batchsize_seqlen, _ = x.shape
        scores = self.gate(x)
        # Llama4 used sigmoid and OLMOE used softmax, we use softmax for now
        # because we need to calculate lb-loss
        # scores = torch.sigmoid(scores.to(torch.float32)).to(x.dtype)
        scores = torch.softmax(scores, dim=1).clamp(min=1e-10, max=1 - 1e-10)
        logits = torch.log(scores / (1 - scores)).reshape(
            batch_size, seq_len, self.num_experts
        )
        rz_loss = torch.logsumexp(logits, dim=[1, 2]).mean()
        eager_assert(scores.shape, (batchsize_seqlen, self.num_experts))

        top_scores, self.selected_experts_indices = torch.topk(
            scores, k=self.experts_per_token, dim=1
        )
        eager_assert(top_scores.shape, (batchsize_seqlen, self.experts_per_token))
        eager_assert(
            self.selected_experts_indices.shape,
            (batchsize_seqlen, self.experts_per_token),
        )

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            self.selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )
        eager_assert(num_tokens_per_expert.shape, (self.num_experts,))
        # group token ids that are assigned to the same expert together
        # sorted it so that we can split the list easily
        token_indices_experts_sorted = torch.argsort(
            self.selected_experts_indices.view(-1), stable=True
        )
        # reorder top_scores based on the sorted token indices
        top_scores = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = (
            token_indices_experts_sorted // self.experts_per_token
        )
        eager_assert(
            token_indices_experts_sorted.shape,
            (batchsize_seqlen * self.experts_per_token,),
        )
        eager_assert(top_scores.shape, (batchsize_seqlen * self.experts_per_token,))
        eager_assert(
            token_indices_experts_sorted.shape,
            (batchsize_seqlen * self.experts_per_token,),
        )

        return top_scores, token_indices_experts_sorted, num_tokens_per_expert, rz_loss


class MoE(nn.Module):
    """This class implements the moe layer which is Mixture of Experts. Mixture of Experts
    typically consists of a set of expert networks, alongside with a router, which directs input tokens
    to the appropriate experts. See more details in https://arxiv.org/pdf/2407.06204.

    Args:
        experts (nn.Module): experts module.
        router (nn.Module): router module.
        shared_expert (Optional[nn.Module]): shared expert module. Default is None.
    """

    def __init__(
        self,
        *,
        experts: nn.Module,
        router: nn.Module,
        shared_expert: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.experts = experts
        self.router = router
        self.shared_expert = shared_expert
        self.use_grouped_mm = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, slen, dim)``.
        """
        batch_size, max_seq_len, dim = x.shape
        (
            top_scores,
            token_indices,
            num_tokens_per_expert,
            rz_loss,
        ) = self.router(
            x.reshape(batch_size * max_seq_len, dim), batch_size, max_seq_len
        )

        routed_input = x.view(-1, dim)[token_indices]

        # TODO: routed_input is weighted by top_scores, didn't find reference for this though
        routed_input = routed_input * top_scores.reshape(-1, 1)

        routed_output, expert_avg_top_scores = self.experts(
            routed_input, num_tokens_per_expert, top_scores
        )

        out = torch.zeros(
            (batch_size * max_seq_len, dim), device=x.device, dtype=routed_output.dtype
        )

        out.index_add_(dim=0, index=token_indices, source=routed_output)

        out = out.reshape(batch_size, max_seq_len, dim)
        # calcualte lb-loss https://arxiv.org/abs/2409.02060
        expert_distribution = num_tokens_per_expert / num_tokens_per_expert.sum()
        lb_loss = (
            torch.sum(expert_distribution * expert_avg_top_scores)
            * self.experts.num_experts
        )

        return (
            out,
            {"lb_loss": lb_loss, "rz_loss": rz_loss},
            {"num_tokens_per_expert": num_tokens_per_expert},
        )


class GroupedExperts(nn.Module):
    """
    Experts for MoE module. For now we use the gated MLP experts, which are the default MLP for Llama2.
    We can also try to update them to SwiGLU later.
    """

    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,  # traiditional 4*dim
        multiple_of: int,
        num_experts: int = 1,
        inference_mode: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim
        self.w13 = nn.Parameter(torch.empty(num_experts, dim, 2 * hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.use_grouped_mm = False
        self.reset_parameters()
        self.inference_mode = inference_mode

    def reset_parameters(self) -> None:
        # Default initialization used by torch.nn.Linear
        nn.init.kaiming_uniform_(self.w13, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def _lengths_to_expert_indices(
        self, lengths: torch.Tensor, T: int, device
    ) -> torch.Tensor:
        starts = torch.cumsum(lengths, dim=0) - lengths
        diffs = torch.zeros(T + 1, dtype=torch.long, device=device)
        diffs.scatter_add_(
            0, starts, torch.ones_like(starts, dtype=torch.long, device=device)
        )
        return diffs[:T].cumsum(0) - 1

    def _forward_autotune_compatible(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Optimized forward pass that replaces the expert loop with vectorized operations.
        """
        T = x.size(0)
        device = x.device
        if T == 0:
            out = x
            expert_avg_top_scores = torch.zeros_like(
                num_tokens_per_expert, dtype=top_scores.dtype, device=device
            )
            return out, expert_avg_top_scores

        lengths = num_tokens_per_expert.to(torch.long)
        E = lengths.numel()

        expert_indices = self._lengths_to_expert_indices(lengths, T, device)

        w13_per_token = torch.index_select(self.w13, 0, expert_indices)
        w2_per_token = torch.index_select(self.w2, 0, expert_indices)

        intermediate = torch.einsum("td,tdh->th", x, w13_per_token)
        x1, x3 = torch.chunk(intermediate, 2, dim=-1)
        h = F.silu(x1) * x3
        out = torch.einsum("th,thd->td", h, w2_per_token)

        sums = torch.zeros(E, dtype=torch.float32, device=device)
        sums.scatter_add_(0, expert_indices, top_scores.to(torch.float32))
        denom = lengths.clamp_min(1).to(torch.float32)
        expert_avg_top_scores = (sums / denom).to(top_scores.dtype)

        return out, expert_avg_top_scores

    def _forward_no_grouped_mm(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.split(
            x,
            split_size_or_sections=num_tokens_per_expert.tolist(),
            dim=0,
        )
        top_scores = torch.split(
            top_scores,
            split_size_or_sections=num_tokens_per_expert.tolist(),
            dim=0,
        )
        eager_assert(len(x), self.num_experts)
        out_experts_splits = []
        expert_avg_top_scores = []
        for expert_idx, (x_expert, top_score_expert) in enumerate(zip(x, top_scores)):
            w13, w2 = (
                self.w13[expert_idx],
                self.w2[expert_idx],
            )
            x1, x3 = torch.chunk(torch.matmul(x_expert, w13), 2, dim=-1)
            h = F.silu(x1) * x3
            h = torch.matmul(h, w2)

            out_experts_splits.append(h)
            avg_score = top_score_expert.mean()
            # got nan when top_score_expert is empty, impute with 0.0
            avg_score = torch.nan_to_num(avg_score, nan=0.0)
            expert_avg_top_scores.append(avg_score)
        out = torch.cat(out_experts_splits, dim=0)
        expert_avg_top_scores = torch.stack(expert_avg_top_scores)
        return out, expert_avg_top_scores

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        top_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor with shape ``(bsz * seq_len * experts_per_token, dim)``
            num_tokens_per_expert (torch.Tensor): Tensor with shape ``(num_experts,)``
                enumerating the number of tokens each expert receives

        Returns:
            torch.Tensor: tensor with shape ``(bsz * seq_len * experts_per_token, dim)``
        """
        if self.inference_mode:
            return self._forward_autotune_compatible(
                x, num_tokens_per_expert, top_scores
            )
        return self._forward_no_grouped_mm(x, num_tokens_per_expert, top_scores)
