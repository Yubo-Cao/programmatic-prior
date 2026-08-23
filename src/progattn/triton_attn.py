"""Causal attention with a programmatic logit prior, as a single Triton kernel.

The FlexAttention implementation in :mod:`progattn.model` is correct but pays an
order of magnitude per layer, and the cost is not the arithmetic in ``score_mod``.
It is autograd: ``score_mod`` closes over ``alpha()``, a grad-carrying tensor, so
FlexAttention has to differentiate through a captured value and reduce a gradient
across the whole score matrix. Two measurements pin it down - the ``incorrect``
arm runs two extra integer modulos per score element for free, and per-step cost
tracks the number of prior-carrying layers rather than program width.

Here the bonus is folded into the score inside the flash loop, where it costs a
handful of registers, and the gradient of a per-head scalar is accumulated in
registers and committed with one atomic per block instead of one per element.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import triton
import triton.language as tl


@triton.jit
def _preferred(
    q_pos,
    k_pos,
    ptype,
    param,
    head_id,
    layer,
    control_seed,
    INCORRECT: tl.constexpr,
):
    """The program DSL predicate, evaluated per (query, key) pair.

    Mirrors ``programs.preferred_edges`` exactly, including the causal rotation
    that the incorrect-prior control uses to keep the number of preferred edges
    per row identical while destroying which edges they are.
    """
    kk = k_pos
    if INCORRECT:
        span = q_pos + 1
        positive_q = tl.maximum(q_pos, 1)
        raw = control_seed + 131 * head_id + 17 * layer
        shift = tl.where(q_pos > 0, 1 + raw % positive_q, 0)
        kk = (k_pos + shift) % span
    first = (ptype == 1) & (kk == 0)
    self_edge = (ptype == 2) & (kk == q_pos)
    source = tl.maximum(q_pos - param, 0)
    previous = (ptype == 3) & (kk == source)
    left = tl.maximum(q_pos - param + 1, 0)
    local = (ptype == 4) & (kk >= left) & (kk <= q_pos)
    return first | self_edge | previous | local


@triton.jit
def _fwd(
    Q,
    K,
    V,
    Beta,
    Types,
    Params,
    Out,
    L,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lm,
    H,
    T,
    sm_scale,
    layer,
    control_seed,
    APPLY_PRIOR: tl.constexpr,
    INCORRECT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    scale = sm_scale.to(tl.float32)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_valid = offs_m < T

    q_ptrs = (
        Q
        + off_b * stride_qb
        + off_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=m_valid[:, None], other=0.0)

    beta = tl.load(Beta + off_h).to(tl.float32) if APPLY_PRIOR else 0.0
    ptype = tl.load(Types + off_h) if APPLY_PRIOR else 0
    param = tl.load(Params + off_h) if APPLY_PRIOR else 0

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    hi = tl.minimum((start_m + 1) * BLOCK_M, T)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < T
        k_ptrs = (
            K
            + off_b * stride_kb
            + off_h * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        v_ptrs = (
            V
            + off_b * stride_vb
            + off_h * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        k = tl.load(k_ptrs, mask=n_valid[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_valid[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        if APPLY_PRIOR:
            pref = _preferred(
                offs_m[:, None],
                offs_n[None, :],
                ptype,
                param,
                off_h,
                layer,
                control_seed,
                INCORRECT,
            )
            qk = qk + tl.where(pref, beta, 0.0).to(tl.float32)
        keep = (
            (offs_m[:, None] >= offs_n[None, :]) & n_valid[None, :] & m_valid[:, None]
        )
        qk = tl.where(keep, qk, float("-inf")).to(tl.float32)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij = tl.where(m_ij == float("-inf"), 0.0, m_ij).to(tl.float32)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        rescale = tl.exp(m_i - m_ij)
        l_i = l_i * rescale + l_ij
        acc = acc * rescale[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_ij

    l_safe = tl.where(l_i == 0.0, 1.0, l_i).to(tl.float32)
    acc = acc / l_safe[:, None]
    lse = m_i + tl.log(l_safe)

    o_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=m_valid[:, None])
    tl.store(
        L + off_b * stride_lb + off_h * stride_lh + offs_m * stride_lm,
        lse,
        mask=m_valid,
    )


@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    stride_db,
    stride_dh,
    stride_dm,
    H,
    T,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_valid = offs_m < T
    base = off_b * stride_ob + off_h * stride_oh
    o = tl.load(
        Out + base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=m_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO + base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
        mask=m_valid[:, None],
        other=0.0,
    ).to(tl.float32)
    tl.store(
        Delta + off_b * stride_db + off_h * stride_dh + offs_m * stride_dm,
        tl.sum(o * do, 1),
        mask=m_valid,
    )


@triton.jit
def _bwd_dkdv(
    Q,
    K,
    V,
    DO,
    DK,
    DV,
    L,
    Delta,
    Beta,
    Types,
    Params,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_lb,
    stride_lh,
    stride_lm,
    H,
    T,
    sm_scale,
    layer,
    control_seed,
    APPLY_PRIOR: tl.constexpr,
    INCORRECT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    scale = sm_scale.to(tl.float32)
    base = off_b * stride_qb + off_h * stride_qh
    lbase = off_b * stride_lb + off_h * stride_lh

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    n_valid = offs_n < T

    k = tl.load(
        K + base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=n_valid[:, None],
        other=0.0,
    )
    v = tl.load(
        V + base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=n_valid[:, None],
        other=0.0,
    )
    dk = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

    beta = tl.load(Beta + off_h).to(tl.float32) if APPLY_PRIOR else 0.0
    ptype = tl.load(Types + off_h) if APPLY_PRIOR else 0
    param = tl.load(Params + off_h) if APPLY_PRIOR else 0

    lo = start_n * BLOCK_N
    for start_m in range(lo, T, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        m_valid = offs_m < T
        q = tl.load(
            Q + base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=m_valid[:, None],
            other=0.0,
        )
        do = tl.load(
            DO + base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=m_valid[:, None],
            other=0.0,
        )
        lse = tl.load(L + lbase + offs_m * stride_lm, mask=m_valid, other=0.0)
        delta = tl.load(Delta + lbase + offs_m * stride_lm, mask=m_valid, other=0.0)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        if APPLY_PRIOR:
            pref = _preferred(
                offs_m[:, None],
                offs_n[None, :],
                ptype,
                param,
                off_h,
                layer,
                control_seed,
                INCORRECT,
            )
            qk = qk + tl.where(pref, beta, 0.0).to(tl.float32)
        keep = (
            (offs_m[:, None] >= offs_n[None, :]) & n_valid[None, :] & m_valid[:, None]
        )
        p = tl.where(keep, tl.exp(qk - lse[:, None]), 0.0).to(tl.float32)

        dv += tl.dot(tl.trans(p).to(do.dtype), do).to(tl.float32)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds).to(q.dtype), q).to(tl.float32) * scale

    tl.store(
        DK + base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        dk.to(DK.dtype.element_ty),
        mask=n_valid[:, None],
    )
    tl.store(
        DV + base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        dv.to(DV.dtype.element_ty),
        mask=n_valid[:, None],
    )


@triton.jit
def _bwd_dq(
    Q,
    K,
    V,
    DO,
    DQ,
    DBeta,
    L,
    Delta,
    Beta,
    Types,
    Params,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_lb,
    stride_lh,
    stride_lm,
    H,
    T,
    sm_scale,
    layer,
    control_seed,
    APPLY_PRIOR: tl.constexpr,
    INCORRECT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    scale = sm_scale.to(tl.float32)
    base = off_b * stride_qb + off_h * stride_qh
    lbase = off_b * stride_lb + off_h * stride_lh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    m_valid = offs_m < T

    q = tl.load(
        Q + base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=m_valid[:, None],
        other=0.0,
    )
    do = tl.load(
        DO + base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        mask=m_valid[:, None],
        other=0.0,
    )
    lse = tl.load(L + lbase + offs_m * stride_lm, mask=m_valid, other=0.0)
    delta = tl.load(Delta + lbase + offs_m * stride_lm, mask=m_valid, other=0.0)

    beta = tl.load(Beta + off_h).to(tl.float32) if APPLY_PRIOR else 0.0
    ptype = tl.load(Types + off_h) if APPLY_PRIOR else 0
    param = tl.load(Params + off_h) if APPLY_PRIOR else 0

    dq = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    dbeta = tl.zeros((1,), tl.float32)

    hi = tl.minimum((start_m + 1) * BLOCK_M, T)
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_valid = offs_n < T
        k = tl.load(
            K + base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=n_valid[:, None],
            other=0.0,
        )
        v = tl.load(
            V + base + offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qd,
            mask=n_valid[:, None],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        if APPLY_PRIOR:
            pref = _preferred(
                offs_m[:, None],
                offs_n[None, :],
                ptype,
                param,
                off_h,
                layer,
                control_seed,
                INCORRECT,
            )
            qk = qk + tl.where(pref, beta, 0.0).to(tl.float32)
        else:
            pref = offs_m[:, None] < 0
        keep = (
            (offs_m[:, None] >= offs_n[None, :]) & n_valid[None, :] & m_valid[:, None]
        )
        p = tl.where(keep, tl.exp(qk - lse[:, None]), 0.0).to(tl.float32)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k).to(tl.float32) * scale
        if APPLY_PRIOR:
            # beta is constant on the preferred set, so its gradient is just the
            # sum of the score gradients there. Accumulating in a register and
            # committing once per block is what keeps this off the critical path.
            dbeta += tl.sum(tl.where(pref & keep, ds, 0.0).to(tl.float32))

    tl.store(
        DQ + base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
        dq.to(DQ.dtype.element_ty),
        mask=m_valid[:, None],
    )
    if APPLY_PRIOR:
        tl.atomic_add(DBeta + off_h + tl.arange(0, 1), dbeta)


def _blocks(sequence_length: int) -> tuple[int, int]:
    size = 1 << max(4, min(sequence_length - 1, 63)).bit_length()
    block = max(16, min(64, size))
    return block, block


class _ProgramAttention(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        types: torch.Tensor,
        params: torch.Tensor,
        layer: int,
        control_seed: int,
        incorrect: bool,
        apply_prior: bool,
    ) -> torch.Tensor:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        batch, heads, length, dim = q.shape
        if dim & (dim - 1):
            raise ValueError(f"head dim must be a power of two, got {dim}")
        block_m, block_n = _blocks(length)
        out = torch.empty_like(q)
        lse = torch.empty((batch, heads, length), device=q.device, dtype=torch.float32)
        beta32 = beta.to(torch.float32).contiguous()
        types32 = types.to(torch.int32).contiguous()
        params32 = params.to(torch.int32).contiguous()
        scale = 1.0 / math.sqrt(dim)
        _fwd[(triton.cdiv(length, block_m), batch * heads)](
            q,
            k,
            v,
            beta32,
            types32,
            params32,
            out,
            lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            heads,
            length,
            scale,
            layer,
            control_seed,
            APPLY_PRIOR=apply_prior,
            INCORRECT=incorrect,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=dim,
        )
        ctx.save_for_backward(q, k, v, out, lse, beta32, types32, params32)
        ctx.layer = layer
        ctx.control_seed = control_seed
        ctx.incorrect = incorrect
        ctx.apply_prior = apply_prior
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[Any, ...]:
        q, k, v, out, lse, beta32, types32, params32 = ctx.saved_tensors
        grad_out = grad_outputs[0].contiguous()
        batch, heads, length, dim = q.shape
        block_m, block_n = _blocks(length)
        delta = torch.empty_like(lse)
        _bwd_preprocess[(triton.cdiv(length, block_m), batch * heads)](
            out,
            grad_out,
            delta,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            delta.stride(0),
            delta.stride(1),
            delta.stride(2),
            heads,
            length,
            BLOCK_M=block_m,
            BLOCK_D=dim,
        )
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dbeta = torch.zeros_like(beta32)
        scale = 1.0 / math.sqrt(dim)
        _bwd_dkdv[(triton.cdiv(length, block_n), batch * heads)](
            q,
            k,
            v,
            grad_out,
            dk,
            dv,
            lse,
            delta,
            beta32,
            types32,
            params32,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            heads,
            length,
            scale,
            ctx.layer,
            ctx.control_seed,
            APPLY_PRIOR=ctx.apply_prior,
            INCORRECT=ctx.incorrect,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=dim,
        )
        _bwd_dq[(triton.cdiv(length, block_m), batch * heads)](
            q,
            k,
            v,
            grad_out,
            dq,
            dbeta,
            lse,
            delta,
            beta32,
            types32,
            params32,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            heads,
            length,
            scale,
            ctx.layer,
            ctx.control_seed,
            APPLY_PRIOR=ctx.apply_prior,
            INCORRECT=ctx.incorrect,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=dim,
        )
        grad_beta = dbeta if ctx.apply_prior else None
        return dq, dk, dv, grad_beta, None, None, None, None, None, None


def program_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    types: torch.Tensor,
    params: torch.Tensor,
    *,
    layer: int,
    control_seed: int = 1729,
    incorrect: bool = False,
    apply_prior: bool = True,
) -> torch.Tensor:
    """Causal attention over ``[B, H, T, D]`` with a per-head bonus on program edges."""
    return cast(
        torch.Tensor,
        _ProgramAttention.apply(
            q, k, v, beta, types, params, layer, control_seed, incorrect, apply_prior
        ),
    )
