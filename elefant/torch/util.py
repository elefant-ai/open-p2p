import torch
import os
from typing import Optional
from contextlib import contextmanager
import logging, time
from typing import Tuple
import torch.nn as nn


ELEFANT_WANDB_DIR = "/tmp/elefant_wandb"


def pytorch_setup(set_seed: bool = False):
    if set_seed:
        torch.manual_seed(0)
        # Make pytorch deterministic.
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

    torch.set_float32_matmul_precision("high")

    # Keep wandb logs out of NFS.
    os.makedirs("/tmp/elefant_wandb", exist_ok=True)
    os.environ["WANDB_DIR"] = "/tmp/elefant_wandb"
    # Cache complied models for faster inference startup.
    # https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_compiler/inductor_cache"
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"
    os.makedirs("/tmp/torch_compiler/inductor_cache", exist_ok=True)


def eager_assert(left_cond, right_cond):
    if not torch.compiler.is_compiling():
        assert left_cond == right_cond, "Eager assert failed: {} != {}".format(
            left_cond, right_cond
        )


def cross_entropy_to_perplexity(cross_entropy: torch.Tensor) -> torch.Tensor:
    return torch.exp(cross_entropy)


def cross_entropy_to_bits_per_dim(cross_entropy: torch.Tensor) -> torch.Tensor:
    return cross_entropy / torch.log(
        torch.tensor(2, dtype=cross_entropy.dtype, device=cross_entropy.device)
    )


@contextmanager
def log_time(label: str):
    """Log wall-clock time (s) for a code block."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logging.info(f"[TIMER] {label}: {elapsed:0.2f}s")


def _sample_from_logits_gpu(
    action_logits: torch.Tensor,
    temperature: float,
    u_scalar: Optional[torch.Tensor] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """
    Returns a sampled index
    - If u_scalar is None, uses Gumbel-max
    - If u_scalar is provided (scalar in [0,1)), uses inverse-CDF with softmax

    We use this function compared to using torch.dist.Categorical due to it being
    faster in sampling on GPU compared to regular torch dist Categorical
    If top_p is set in (0,1), applies nucleus sampling after temperature
    """
    logits = action_logits.to(torch.float32)
    tau = max(float(temperature), 1e-8)

    scaled_logits = logits if tau == 1.0 else (logits / tau)
    apply_top_p = (top_p is not None) and (0.0 < float(top_p) < 1.0)

    def apply_top_p_mask_from_probs(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        # keep minimal set whose cumulative prob >= top_p
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        to_remove = cumsum > top_p
        to_remove[..., 1:] = to_remove[..., :-1].clone()
        to_remove[..., 0] = False

        # Scatter removal mask back to original order
        remove_mask = torch.zeros_like(to_remove, dtype=torch.bool)
        remove_mask.scatter_(dim=-1, index=sorted_idx, src=to_remove)
        return ~remove_mask  # keep mask

    if u_scalar is None:
        if apply_top_p:
            probs = torch.softmax(scaled_logits, dim=-1)
            keep_mask = apply_top_p_mask_from_probs(probs, float(top_p))
            scaled_logits = scaled_logits.masked_fill(~keep_mask, float("-inf"))

        gumbel = -torch.log(-torch.log(torch.rand_like(scaled_logits)))
        scores = scaled_logits + gumbel
        idx = scores.argmax(dim=-1, keepdim=True)
        return idx.to(torch.long)

    probs = torch.softmax(scaled_logits, dim=-1)
    if apply_top_p:
        keep_mask = apply_top_p_mask_from_probs(probs, float(top_p))
        kept = probs * keep_mask.to(probs.dtype)
        probs = kept / kept.sum(dim=-1, keepdim=True)

    cdf = torch.cumsum(probs, dim=-1)
    cdf[..., -1] = 1.0  # ensure last entry is exactly 1
    u = u_scalar.to(logits.device, dtype=torch.float32)
    u = torch.clamp_min(u, 0.0)
    u = torch.where(
        u >= 1.0,
        torch.nextafter(torch.ones_like(u), torch.zeros_like(u)),
        u,
    )
    u_b = u.view(1, 1, 1)

    # Find first index where CDF >= u
    mask = cdf >= u_b
    # Given u <= 1 - 1e-7 and cdf[..., -1] == 1, there is always at least one True
    idx = mask.to(torch.int64).argmax(dim=-1, keepdim=True)
    return idx.to(torch.long)


def count_model_parameters(model: nn.Module) -> Tuple[int, int]:
    total_params = 0
    expert_params = 0

    # Iterate over all named parameters in the model
    for name, param in model.named_parameters():
        # Skip parameters that are not set to be trained
        if not param.requires_grad:
            continue

        # Get the number of elements in the current parameter tensor
        num_params = param.numel()
        total_params += num_params

        # Check if the parameter's name indicates it's part of an expert layer
        if "ffn" in name:
            expert_params += num_params

    return total_params, expert_params
