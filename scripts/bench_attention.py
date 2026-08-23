"""Correctness and throughput for the Triton programmatic-prior attention kernel.

Correctness is checked against the dense float32 path the model already uses for
discovery, which is the same arithmetic the FlexAttention arms are meant to
implement. The beta gradient additionally gets a finite-difference check, because
a per-head scalar summed over the whole score matrix is exactly the kind of term
that can be wrong by a constant factor and still look plausible.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from typing import cast

import torch

from progattn.programs import ProgramType, dense_program_mask
from progattn.triton_attn import program_attention


def reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    types: torch.Tensor,
    params: torch.Tensor,
    *,
    layer: int,
    control_seed: int,
    incorrect: bool,
    apply_prior: bool,
) -> torch.Tensor:
    length = q.size(-2)
    work = torch.float64 if q.dtype == torch.float64 else torch.float32
    scores = q.to(work) @ k.to(work).transpose(-2, -1) / math.sqrt(q.size(-1))
    causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
    if apply_prior:
        mask = dense_program_mask(
            types,
            params,
            length,
            layer=layer,
            incorrect=incorrect,
            control_seed=control_seed,
        )
        scores = scores + beta.to(work)[None, :, None, None] * mask[None]
    scores = scores.masked_fill(~causal, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v.to(work)


def make_inputs(
    batch: int,
    heads: int,
    length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (batch, heads, length, dim)
    return tuple(
        torch.randn(
            shape, device=device, dtype=dtype, generator=generator
        ).requires_grad_(True)
        for _ in range(3)
    )


def check(args: argparse.Namespace) -> bool:
    device = torch.device("cuda")
    dtype = torch.float32
    batch, heads, length = 2, 6, 80
    types = torch.zeros(heads, dtype=torch.int32, device=device)
    params = torch.zeros(heads, dtype=torch.int32, device=device)
    plan = [
        (ProgramType.LOCAL_WINDOW, 4),
        (ProgramType.LOCAL_WINDOW, 8),
        (ProgramType.PREVIOUS_K, 1),
        (ProgramType.FIRST_TOKEN, 0),
        (ProgramType.SELF, 0),
        (ProgramType.NONE, 0),
    ]
    for head in range(heads):
        program, parameter = plan[head % len(plan)]
        types[head] = int(program)
        params[head] = parameter

    ok = True
    for incorrect in (False, True):
        q, k, v = make_inputs(batch, heads, length, args.dim, device, dtype, 7)
        beta = torch.full((heads,), 4.0, device=device, dtype=torch.float32)
        beta.requires_grad_(True)
        grad_seed = torch.randn(
            (batch, heads, length, args.dim),
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(11),
        )

        out = program_attention(
            q,
            k,
            v,
            beta,
            types,
            params,
            layer=args.layer,
            control_seed=args.control_seed,
            incorrect=incorrect,
            apply_prior=True,
        )
        out.backward(grad_seed)
        got = [out.detach(), q.grad, k.grad, v.grad, beta.grad]

        q2, k2, v2 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        beta2 = beta.detach().clone().requires_grad_(True)
        ref = reference(
            q2,
            k2,
            v2,
            beta2,
            types,
            params,
            layer=args.layer,
            control_seed=args.control_seed,
            incorrect=incorrect,
            apply_prior=True,
        )
        ref.backward(grad_seed)
        want = [ref.detach(), q2.grad, k2.grad, v2.grad, beta2.grad]

        label = "incorrect" if incorrect else "matched"
        for name, a, b in zip(
            ("out", "dq", "dk", "dv", "dbeta"), got, want, strict=True
        ):
            assert a is not None and b is not None
            delta = (a - b).abs().max().item()
            scale = b.abs().max().item()
            rel = delta / max(scale, 1e-12)
            status = "ok " if rel < 2e-3 else "FAIL"
            if rel >= 2e-3:
                ok = False
            print(
                f"  [{label:9s}] {name:6s} {status} max_abs {delta:.3e} rel {rel:.3e}"
            )

        # Finite difference on a single head's beta, which pins the scale of the
        # atomic reduction independently of the analytic derivation.
        head = 0
        epsilon = 1e-3
        q64, k64, v64 = (t.detach().double() for t in (q, k, v))
        seed64 = grad_seed.double()
        with torch.no_grad():
            losses = []
            for sign in (1.0, -1.0):
                shifted = beta.detach().double().clone()
                shifted[head] += sign * epsilon
                value = reference(
                    q64,
                    k64,
                    v64,
                    shifted,
                    types,
                    params,
                    layer=args.layer,
                    control_seed=args.control_seed,
                    incorrect=incorrect,
                    apply_prior=True,
                )
                losses.append((value * seed64).sum().item())
        numeric = (losses[0] - losses[1]) / (2 * epsilon)
        assert beta.grad is not None
        analytic = float(beta.grad[head].item())
        rel = abs(numeric - analytic) / max(abs(numeric), 1e-9)
        status = "ok " if rel < 5e-3 else "FAIL"
        if rel >= 5e-3:
            ok = False
        print(
            f"  [{label:9s}] dbeta[{head}] finite-diff {status} "
            f"analytic {analytic:.6f} numeric {numeric:.6f} rel {rel:.3e}"
        )
    return ok


def timed(fn: Callable[[], None], iterations: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iterations


def bench(args: argparse.Namespace) -> None:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    from progattn.model import causal_mask, noop_score_mod
    from progattn.programs import make_score_mod

    device = torch.device("cuda")
    dtype = torch.bfloat16
    heads, length, dim = args.heads, args.length, args.dim
    types = torch.zeros(heads, dtype=torch.int32, device=device)
    params = torch.zeros(heads, dtype=torch.int32, device=device)
    for head in range(0, heads, 2):
        types[head] = int(ProgramType.LOCAL_WINDOW)
        params[head] = 8
    q, k, v = make_inputs(args.batch, heads, length, dim, device, dtype, 3)
    grad_seed = torch.randn_like(q)
    beta = torch.full((heads,), 4.0, device=device, dtype=torch.float32).requires_grad_(
        True
    )

    block_mask = create_block_mask(
        causal_mask, B=None, H=None, Q_LEN=length, KV_LEN=length, device=device
    )
    flex = torch.compile(flex_attention, dynamic=False)

    def clear() -> None:
        for t in (q, k, v):
            t.grad = None
        beta.grad = None

    def run_sdpa() -> None:
        clear()
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.backward(grad_seed)

    def run_flex_noop() -> None:
        clear()
        out = cast(
            torch.Tensor, flex(q, k, v, score_mod=noop_score_mod, block_mask=block_mask)
        )
        out.backward(grad_seed)

    def run_flex_prior() -> None:
        clear()
        mod = make_score_mod(
            types,
            params,
            beta,
            layer=args.layer,
            incorrect=False,
            control_seed=args.control_seed,
        )
        out = cast(torch.Tensor, flex(q, k, v, score_mod=mod, block_mask=block_mask))
        out.backward(grad_seed)

    def run_triton() -> None:
        clear()
        out = program_attention(
            q,
            k,
            v,
            beta,
            types,
            params,
            layer=args.layer,
            control_seed=args.control_seed,
            incorrect=False,
            apply_prior=True,
        )
        out.backward(grad_seed)

    cases = [
        ("sdpa flash (no prior)", run_sdpa),
        ("flex noop (no prior)", run_flex_noop),
        ("flex + score_mod prior", run_flex_prior),
        ("triton prior kernel", run_triton),
    ]
    print(f"\n  shape B={args.batch} H={heads} T={length} D={dim} dtype={dtype}")
    baseline = None
    for name, fn in cases:
        try:
            ms = timed(fn, args.iterations)
        except Exception as error:
            print(f"  {name:24s} FAILED {type(error).__name__}: {error}")
            continue
        if baseline is None:
            baseline = ms
        print(f"  {name:24s} {ms:8.3f} ms/iter   {baseline / ms:5.2f}x vs flash")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--control-seed", type=int, default=1729)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--skip-check", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    args = parser.parse_args()
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name(0)}")
    ok = True
    if not args.skip_check:
        print("\ncorrectness vs dense float32 reference:")
        ok = check(args)
        print(f"\n  correctness: {'PASS' if ok else 'FAIL'}")
    if not args.skip_bench:
        bench(args)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
