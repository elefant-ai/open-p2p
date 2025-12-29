import math
import torch
from elefant.torch import _sample_from_logits_gpu


@torch.no_grad()
def test_against_categorical():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(123)

    B, T, V = 64, 16, 7
    reps = 300
    N = B * T * reps

    logits = (torch.randn(B, T, V, device=device) * 1.7) + 0.2
    probs = torch.softmax(logits, dim=-1)
    expected_frac = probs.sum(dim=(0, 1)) / (B * T)

    def sample_counts(temp, reps, u=None, check=False):
        idxs = [
            _sample_from_logits_gpu(logits, temperature=temp, u_scalar=u)
            for _ in range(reps)
        ]
        if check:
            i0 = idxs[0]
            assert i0.shape == (B, T, 1) and i0.dtype == torch.long
            assert ((i0 >= 0) & (i0 < V)).all()
        return torch.bincount(torch.stack(idxs).view(-1), minlength=V)

    # 1) Gumbel-max vs expected and vs Categorical at T=1
    counts_1 = sample_counts(1.0, reps, check=True)
    emp_1 = counts_1.float() / N
    err = (emp_1 - expected_frac).abs().max().item()
    assert err < 0.015, f"Gumbel T=1 vs expected: {err:.5f}"

    dist = torch.distributions.Categorical(logits=logits)
    idx_cat = dist.sample((reps,))  # (reps, B, T)
    emp_cat = torch.bincount(idx_cat.reshape(-1), minlength=V).float() / N
    diff = (emp_1 - emp_cat).abs().max().item()
    assert diff < 0.015, f"Custom vs Categorical: {diff:.5f}"

    # 2) Gumbel discrete should be temperature-invariant
    se = math.sqrt(0.5 / float(N))
    tol_same = max(6.0 * se, 1e-3)
    for tau in (0.5, 2.0):
        emp_tau = sample_counts(tau, reps).float() / N
        err_tau = (emp_tau - expected_frac).abs().max().item()
        delta = (emp_tau - emp_1).abs().max().item()
        assert err_tau < 0.015, f"T={tau} vs expected: {err_tau:.5f}"
        assert delta < tol_same, f"T={tau} vs T=1 delta={delta:.5f}, tol={tol_same:.5f}"

    # 3) Inverse-CDF branch matches searchsorted
    for tau in (0.5, 1.0, 2.0):
        cdf = torch.cumsum(torch.softmax(logits / tau, dim=-1), dim=-1).contiguous()
        for _ in range(50):
            u = torch.rand((), device=device).clamp_(0.0, 1.0 - 1e-7)
            idx_custom = _sample_from_logits_gpu(logits, temperature=tau, u_scalar=u)
            u_vals = u.expand_as(cdf[..., :1]).contiguous()
            idx_ref = torch.searchsorted(cdf, u_vals, right=False).clamp(max=V - 1)
            assert torch.equal(idx_custom, idx_ref), (
                f"Inverse-CDF mismatch T={tau}, u={float(u):.6f}"
            )

    # 4) Edge checks for u=0 and u≈1
    idx_u0 = _sample_from_logits_gpu(
        logits, temperature=1.0, u_scalar=torch.tensor(0.0, device=device)
    )
    assert torch.all(idx_u0 == 0), "u=0 should select index 0"

    u_hi = torch.tensor(1.0 - 1e-7, device=device)
    idx_u1m = _sample_from_logits_gpu(logits, temperature=1.0, u_scalar=u_hi)
    cdf_1 = torch.cumsum(torch.softmax(logits, dim=-1), dim=-1)
    idx_ref_hi = torch.searchsorted(
        cdf_1, u_hi.expand_as(cdf_1[..., :1]), right=False
    ).clamp(max=V - 1)
    assert torch.equal(idx_u1m, idx_ref_hi), "u≈1 must match searchsorted(cdf, u)"


@torch.no_grad()
def test_top_p_tiny_selects_argmax():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(123)

    B, T, V = 32, 8, 11
    logits = (torch.randn(B, T, V, device=device) * 1.7) + 0.2
    argmax_idx = logits.argmax(dim=-1, keepdim=True)  # (B,T,1)

    # should collapse to argmax when only top-1 is allowed
    idx_g = _sample_from_logits_gpu(logits, temperature=0.8, u_scalar=None, top_p=1e-6)
    assert idx_g.shape == (B, T, 1) and idx_g.dtype == torch.long
    assert torch.equal(idx_g, argmax_idx), "tiny top_p should pick argmax (Gumbel path)"
