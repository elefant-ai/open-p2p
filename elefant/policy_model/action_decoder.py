"""
Defines a small-ish transformer to map from a single token to an auto-regressive sequence of tokens.
"""

import torch
import torch.nn as nn
from elefant.policy_model.transformer import (
    Transformer,
    TransformerConfig,
    SelfAttention,
)
from elefant.torch import eager_assert
from typing import Callable, Optional
from elefant.policy_model.kv_cache import AccumulatingKVCache
from elefant.policy_model.transformer import TransformerSelfAttentionLayer


class ActionDecoderConfig(TransformerConfig):
    n_action_tokens: int = 1
    input_action_token_dim: int = 256


class ActionDecoder(Transformer):
    def __init__(self, cfg: ActionDecoderConfig):
        super().__init__(config=cfg)

        self.cfg = cfg
        # There will be one more action tokens than actual number of actions.
        # because a^0 is the start token.
        self.n_action_tokens = cfg.n_action_tokens
        self.max_seq_len = self.n_action_tokens
        self.embedding_std = 3.0

        # We need a marker for each action position.
        self.pos_tokens = nn.Parameter(
            torch.empty(self.n_action_tokens, self.cfg.embed_dim, dtype=torch.bfloat16)
        )
        torch.nn.init.normal_(self.pos_tokens, mean=0.0, std=self.embedding_std)

        self.input_proj = nn.Linear(cfg.input_action_token_dim, self.cfg.embed_dim)

        self.construct_transformer_layers()
        self._setup_kv_cache_state()

    def construct_self_attention(self):
        """Override this to modify the self attention block."""
        attention_block = SelfAttention(
            input_dim=self.config.embed_dim,
            embed_size_per_head=self.config.embed_dim // self.config.n_q_head,
            n_q_head=self.config.n_q_head,
            n_kv_head=self.config.n_kv_head,
            dropout=self.config.dropout,
            pos_embedding=self.pos_embeddings,
            is_causal=True,
        )
        return attention_block

    def _setup_kv_cache_state(self):
        self.kv_cache = AccumulatingKVCache(
            batch_size=1,
            device=self.pos_tokens.device,
            dtype=torch.bfloat16,
            num_kv_heads=self.config.n_kv_head,
            embed_size_per_head=self.config.embed_dim // self.config.n_q_head,
        )

        for i, layer in enumerate(self.transformer_layers):
            if not isinstance(layer, TransformerSelfAttentionLayer):
                continue

            sa = layer.self_attention
            sa.kv_cache = self.kv_cache

    def autogressive_sample(
        self,
        input_action_token: torch.Tensor,
        action_sampler: Callable,
        empty_sampled_action_fn: Callable,
        reshape_structured_action_fn: Optional[Callable] = None,
        inference_mode: bool = True,
    ):
        """Autogressively sample from the action decoder, which requires allowing a sample each time."""
        B, T, E = input_action_token.shape
        eager_assert(E, self.cfg.input_action_token_dim)

        sampled_action = empty_sampled_action_fn(B * T)

        input_action_tokens_proj = self.input_proj(input_action_token)
        eager_assert(input_action_tokens_proj.shape, (B, T, self.cfg.embed_dim))
        # The time dimension for this transformer is the action dimension.
        input_action_tokens_proj = input_action_tokens_proj.view(
            B * T, 1, self.cfg.embed_dim
        )
        # Add the position token.
        pos0 = self.pos_tokens[0:1, :].unsqueeze(0)
        eager_assert(pos0.shape, (1, 1, self.cfg.embed_dim))
        input_action_tokens_proj_with_pos = input_action_tokens_proj + pos0

        # We input batch size here instead of _setup_kv_cache_state because
        # the batch size is dynamic (e.g., each batch can have different number of unlabelled datas)
        if inference_mode:
            # inference mode won't chagne shape, so don't pass batch size
            kv_cache_state = self.init_kv_cache_state()
        else:
            kv_cache_state = self.init_kv_cache_state(batch_size=B * T)

        # Use input_pos so RopE positions align with the concatenated (non-cache) pass.
        # First token uses position 0.
        input_pos = torch.arange(0, 1, device=input_action_tokens_proj_with_pos.device)
        y, kv_cache_state, *_ = super().forward(
            input_action_tokens_proj_with_pos,
            input_pos=input_pos,
            kv_cache_state=kv_cache_state,
        )

        eager_assert(y.shape, (B * T, 1, self.cfg.embed_dim))

        for i in range(self.n_action_tokens - 1):
            last_action_in_token = action_sampler(y, i, sampled_action)

            eager_assert(last_action_in_token.shape, (B * T, 1, self.cfg.embed_dim))

            # Add position embedding for the next action token.
            this_pos_token = self.pos_tokens[i + 1 : i + 2, :].unsqueeze(0)
            eager_assert(this_pos_token.shape, (1, 1, self.cfg.embed_dim))
            last_action_in_token = last_action_in_token + this_pos_token

            # Next token should use the next absolute position which equals current KV length.
            # KV cache stores already-rotated keys; we only need to provide the position for new q/k.
            current_kv_len = kv_cache_state[0].k_cache.shape[2]
            next_input_pos = torch.arange(
                current_kv_len, current_kv_len + 1, device=last_action_in_token.device
            )
            y, kv_cache_state, *_ = super().forward(
                last_action_in_token,
                input_pos=next_input_pos,
                kv_cache_state=kv_cache_state,
            )
            eager_assert(y.shape, (B * T, 1, self.cfg.embed_dim))

        if reshape_structured_action_fn is not None:
            sampled_action = reshape_structured_action_fn(sampled_action, B, T)

        return sampled_action

    def forward(
        self, input_action_token: torch.Tensor, action_embeddings_in: torch.Tensor
    ):
        B, T, E = input_action_token.shape
        eager_assert(E, self.cfg.input_action_token_dim)

        # Each timestep has a single action token.
        input_action_tokens_proj = self.input_proj(input_action_token)
        eager_assert(input_action_tokens_proj.shape, (B, T, self.cfg.embed_dim))
        eager_assert(
            action_embeddings_in.shape,
            (B, T, self.n_action_tokens - 1, self.cfg.embed_dim),
        )

        input_action_tokens_proj = input_action_tokens_proj.unsqueeze(2)
        eager_assert(input_action_tokens_proj.shape, (B, T, 1, self.cfg.embed_dim))

        inp_tokens_without_pos = torch.cat(
            [input_action_tokens_proj, action_embeddings_in], dim=2
        )
        eager_assert(
            inp_tokens_without_pos.shape,
            (B, T, self.n_action_tokens, self.cfg.embed_dim),
        )

        inp_tokens_with_pos = inp_tokens_without_pos + self.pos_tokens.unsqueeze(
            0
        ).unsqueeze(0)
        eager_assert(
            inp_tokens_with_pos.shape, (B, T, self.n_action_tokens, self.cfg.embed_dim)
        )

        # Combine B, T dimensions, the "time" dimension in this transformer is the "action" dimension.
        inp_tokens_with_pos = inp_tokens_with_pos.view(
            B * T, self.n_action_tokens, self.cfg.embed_dim
        )

        y, *_ = super().forward(inp_tokens_with_pos)
        eager_assert(y.shape, (B * T, self.n_action_tokens, self.cfg.embed_dim))

        y = y.view(B, T, self.n_action_tokens, self.cfg.embed_dim)
        eager_assert(y.shape, (B, T, self.n_action_tokens, self.cfg.embed_dim))

        # The final token is for inputing the last action.
        # Right now its not used, in future if we allow the model to see past actions it will be.
        y = y[:, :, 0:-1, :]
        eager_assert(y.shape, (B, T, self.n_action_tokens - 1, self.cfg.embed_dim))

        return y
