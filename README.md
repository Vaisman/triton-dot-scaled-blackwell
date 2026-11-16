# A missing scale makes Triton's `tl.dot_scaled` up to ~6x slower on NVIDIA Blackwell

**TL;DR**

- On Blackwell (B200/B300), `tl.dot_scaled` uses the native block-scaled tensor-core MMA (`tcgen05.mma ... kind::mxf8f6f4`) only when **both** scales are given. If one scale is `None` — which the API documents as allowed — the compiler falls back to converting the fp8 operands to bf16 and runs a plain 16-bit MMA (`kind::f16`).
- On a B300, M = N = K = 8192, that fallback is **2.0–5.8x slower per tile configuration** (3.3x best-vs-best) than passing a unit scale (E8M0 `127`, i.e. 1.0), with bit-identical results on exact integer test inputs.
- Nsight Compute on the same configuration: **5.16 ms vs 2.34 ms**, with **82 % vs 44 %** L1/TEX throughput. Profiling points to the bf16 upcast path; the fallback is not DRAM-bound.
- Reported upstream: [triton-lang/triton#12784](https://github.com/triton-lang/triton/issues/12784). Workaround: pass `tl.full(..., 127, tl.uint8)` instead of `None`.

## Background

Blackwell tensor cores can multiply low-precision matrices with **per-block scales** in hardware: for MX formats, every 32 consecutive elements along K share one 8-bit power-of-two scale (E8M0). Triton exposes this as `tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, acc)`, where each scale is "an uint8 tensor, or None".

One such case is having a real scale on one side only, e.g. block-scaled activations with plain fp8 weights.

## What the compiler does

In `lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp`, the pattern that emits the native Blackwell MMA gives up as soon as either scale is missing:

```cpp
// ScaledBlockedToMMAv5::matchAndRewrite
if (dotOp.getAScale() == nullptr || dotOp.getBScale() == nullptr) {
  return failure();
}
```

The dot then goes through `DecomposeScaledBlocked`, which upcasts both fp8 operands to bf16 and emits an ordinary MMA. In PTX this is the difference between

```
tcgen05.mma.cta_group::1.kind::f16 ...                    // fallback (scale = None)
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale ...   // unit scale passed explicitly
```

(A dot where only the RHS scale is present is first transposed, so both one-sided cases end up on the same path.)

## Results

NVIDIA B300 SXM6 (sm_103), Triton `ecd97a63d120`, e4m3 x e4m3 matmul with pointer loads and an fp32 accumulator, M = N = K = 8192, median of 3 runs. The LHS scale is present and varies per 32-element block (E8M0 124..130, i.e. 2^-3..2^3); the RHS scale is either `None` or `tl.full(..., 127)`.

| tile config (BM x BN x BK, warps, stages) | RHS scale `None` (`kind::f16`) | unit RHS scale (`kind::mxf8f6f4`) | speedup |
|---|---|---|---|
| 128x128x128, 4, 3 | 186 TFLOP/s | 1075 TFLOP/s | 5.8x |
| 128x128x128, 8, 3 | 185 | 1028 | 5.6x |
| 128x128x64, 4, 4 | 364 | 723 | 2.0x |
| 128x256x128, 8, 3 | 209 | 1213 | 5.8x |
| **best of the four** | **364** | **1213** | **3.3x** |

Other cases from the same run:

| case | fallback | native | speedup |
|---|---|---|---|
| LHS scale missing instead (same configs, 8192³) | — | — | 1.6–3.3x per config, 2.5x best-vs-best (511 vs 1280) |
| RHS missing, 4096³ (best vs best) | 312 | 974 | 3.1x |
| RHS missing, 2048 x 8192 x 8192 (best vs best) | 318 | 1075 | 3.4x |
| both scales missing vs two unit scales (8192³) | 482 | 1897 | 3.9x |
| plain fp8 `tl.dot` (no scales), for reference | | 1924 | |

## Why it is slower

**Nsight Compute**, same tile config (128x128x64, 4 warps, 4 stages), clocks locked by ncu:

| metric | RHS scale `None` | unit RHS scale |
|---|---|---|
| duration under Nsight Compute | 5.16 ms | 2.34 ms |
| L1/TEX throughput (includes shared memory) | 82 % | 44 % |
| DRAM throughput | 3 % | 7 % |
| compute (SM) throughput | 18 % | 26 % |
| achieved occupancy | 12.5 % | 12.4 % |

The fallback is not DRAM-bound. Profiling points to L1/TEX pressure, consistent with its extra shared-memory traffic for the upcast operands. (Durations under ncu are not comparable with the benchmark numbers above: ncu locks the clocks.)

**SASS** (static instruction counts; each variant at its best config, 128x128x64/4/4 for the fallback and 128x256x128/8/3 for the unit scale):

| | fallback | unit scale |
|---|---|---|
| MMA | 8 x `UTCHMMA` (16-bit) | 4 x `UTCQMMA` (block-scaled) |
| fp8 -> bf16 conversions (`F2FP.BF16.E4M3`) | 128 | 0 |
| shared-memory stores / loads (`STS` / `LDS`) | 50 / 52 | 36 / 16 |

The fallback converts the operands to bf16 and writes them back (RHS to shared memory, LHS to tensor memory) before a 16-bit MMA; the native path feeds fp8 directly to the block-scaled MMA.

## Correctness

Inputs are small integers and scales are powers of two, so each variant is checked **bit for bit** against an fp32 reference that applies the scales; all variants match. A second check with random e4m3 values and random scales (2^-7..2^7) passes `allclose` at each variant's best config (max error ≤ ~1e-5 relative to the largest output). The fallback is a performance problem, not a correctness one.

## Workaround and fix

- **Workaround today:** pass an explicit unit scale for the missing side, e.g. `tl.full((BN, BK // 32), 127, tl.uint8)`. Triton's own `triton_kernels` already does this.
- **Proposed fix:** let `ScaledBlockedToMMAv5` materialize a constant unit scale for a missing side, for the formats the native MMA supports. Discussion: [triton-lang/triton#12784](https://github.com/triton-lang/triton/issues/12784).

## Reproduce

- **Minimal:** `python repro.py` on a Blackwell GPU with Triton built from source. On B300 it prints [`results/repro.out`](results/repro.out):
  ```
  UNIT_RHS_SCALE=False: kind=['f16'] exact=True 186 TFLOP/s
  UNIT_RHS_SCALE=True: kind=['mxf8f6f4'] exact=True 1075 TFLOP/s
  ```
- **Full sweep + profiling:** [`scripts/vm-run-o1.sh`](scripts/vm-run-o1.sh) (steps `check`, `setup`, `build`, `ncu`, `quick`, `full`, `profile`, `pack`) runs [`scripts/bench_o1_fresh.py`](scripts/bench_o1_fresh.py) and [`scripts/ncu_o1_fresh.py`](scripts/ncu_o1_fresh.py) on a fresh VM.

**Environment:** NVIDIA B300 SXM6 (sm_103, driver 580.173.02, 1100 W), Ubuntu 24.04, Triton `ecd97a63d120` built from source (the check above is unchanged on main `46eccb199b`), PyTorch 2.8.0+cu128, ptxas 13.5 (bundled with Triton), Nsight Compute 2025.3.1.

## Caveats

- The kernel is deliberately simple (pointer loads, not TMA-based or persistent), so absolute numbers are below a tuned library; the point is the ratio between the two lowerings.
- e4m3 only; one GPU (B300). An earlier run on B200 (sm_100, older Triton main, unit scales) showed the same pattern.
- cuBLAS `_scaled_mm` with a per-tensor scale reaches ~3000 TFLOP/s on the same GPU; it is a different operation and is only a reference point.

## Files

| path | contents |
|---|---|
| [`repro.py`](repro.py), [`results/repro.out`](results/repro.out) | minimal reproducer and its output on B300 |
| [`results/o1_quick.txt`](results/o1_quick.txt), [`results/o1_full.txt`](results/o1_full.txt) | every configuration, mode and shape, with exactness checks |
| [`results/ncu/`](results/ncu/) | Nsight Compute text reports (`--page details`) |
| [`results/ir/`](results/ir/) | TTGIR, PTX and SASS of the two main variants |
| [`scripts/`](scripts/) | benchmark, profiling and VM setup scripts |
