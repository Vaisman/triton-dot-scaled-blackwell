#!/usr/bin/env python3
"""O1 re-measure on a pinned recent Triton: tl.dot_scaled with ONE missing scale.

Extends bench_o1.py (kept unchanged so the 2026-11-02 numbers stay reproducible) with:
  * modes where the present scale is NON-unit and varies per 32-wide K block (E8M0 124..130 = 2^-3..2^3),
  * the mirror case (LHS scale missing, RHS present),
  * a scale-aware fp32 reference,
  * a random-value sanity check (allclose) next to the exact small-integer check,
  * 3 timing repeats (median, min, max),
  * environment logging and a check that the imported Triton is the pinned checkout (--expect-commit),
  * artifacts for the best config of every mode: TTGIR, PTX, SASS and SASS instruction counts,
  * the expected tcgen05.mma kind per mode, and a NON-ZERO exit status on failures:
      exit 1 = a correctness check failed or the imported Triton is not the expected commit,
      exit 2 = correctness OK but some mode emitted an unexpected mma kind (the finding may have changed).

Modes (fp8 e4m3 operands, fp32 accumulate, C = (A * 2^(SA-127)) @ (B * 2^(SB-127))):
  baseline (unit scales, same as bench_o1.py):
    noscale      dot_scaled(a, None,  b, None)          expected kind::f16 (bf16 fallback)
    lhsonly      dot_scaled(a, sa=127, b, None)         expected kind::f16
    const127     dot_scaled(a, full127, b, full127)     expected mxf8f6f4
    lhs+const    dot_scaled(a, sa=127, b, full127)      expected mxf8f6f4  (the user-level workaround)
    dot          tl.dot(a, b)                           expected f8f6f4
  new (varying, non-unit scale on the present side):
    lhsvar       dot_scaled(a, sa_var, b, None)         expected kind::f16
    lhsvar+const dot_scaled(a, sa_var, b, full127)      expected mxf8f6f4
    rhsvar       dot_scaled(a, None,    b, sb_var)      expected kind::f16
    rhsvar+const dot_scaled(a, full127, b, sb_var)      expected mxf8f6f4

Exactness: inputs are small integers (-3..3) and scales are powers of two in 2^-3..2^3, so every product and partial
sum is a multiple of 2^-3 below 2^21 for K <= 8192 and is exact in fp32 (and the scaled operands are exact in bf16).

    python3 bench_o1_fresh.py --quick --expect-commit <sha> --artifacts art/ --out o1_quick.txt
    python3 bench_o1_fresh.py --expect-commit <sha> --artifacts art/ --out o1_full.txt
"""
import collections
import os
import re
import statistics
import subprocess
import sys

import torch
import triton
import triton.language as tl

# name -> (kernel MODE, which scale buffers: "unit" or "var")
MODES = {
    "noscale": (0, "unit"),
    "lhsonly": (1, "unit"),
    "const127": (4, "unit"),
    "lhs+const": (5, "unit"),
    "dot": (3, "unit"),
    "lhsvar": (1, "var"),
    "lhsvar+const": (5, "var"),
    "rhsvar": (6, "var"),
    "rhsvar+const": (7, "var"),
}
EXPECTED_KIND = {
    "noscale": "f16", "lhsonly": "f16", "lhsvar": "f16", "rhsvar": "f16",
    "const127": "mxf8f6f4", "lhs+const": "mxf8f6f4", "lhsvar+const": "mxf8f6f4", "rhsvar+const": "mxf8f6f4",
    "dot": "f8f6f4",
}
QUICK_MODES = ["lhsonly", "lhs+const", "lhsvar", "lhsvar+const", "rhsvar", "rhsvar+const"]
# Present scale in each kernel MODE (for the reference): (uses SA, uses SB).
USES = {0: (False, False), 1: (True, False), 3: (False, False), 4: (False, False), 5: (True, False),
        6: (False, True), 7: (False, True)}
SASS_OPS = ["UTCHMMA", "UTCQMMA", "UTCOMMA", "F2FP", "STS", "LDS", "LDG", "PRMT", "LDSM", "UTMALDG"]


@triton.jit
def mm_kernel(a_ptr, bt_ptr, sa_ptr, sb_ptr, c_ptr, M, N, K,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rs = tl.arange(0, BK // 32)
    KS = K // 32
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :])       # [BM, BK]
        b = tl.load(bt_ptr + rn[None, :] * K + (k + rk)[:, None])      # [BK, BN], K contiguous
        if MODE == 0:
            acc = tl.dot_scaled(a, None, "e4m3", b, None, "e4m3", acc)
        elif MODE == 1:
            sa = tl.load(sa_ptr + rm[:, None] * KS + (k // 32 + rs)[None, :])
            acc = tl.dot_scaled(a, sa, "e4m3", b, None, "e4m3", acc)
        elif MODE == 3:
            acc = tl.dot(a, b, acc)
        elif MODE == 4:
            sa = tl.full((BM, BK // 32), 127, tl.uint8)
            sb = tl.full((BN, BK // 32), 127, tl.uint8)
            acc = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
        elif MODE == 5:
            sa = tl.load(sa_ptr + rm[:, None] * KS + (k // 32 + rs)[None, :])
            sb = tl.full((BN, BK // 32), 127, tl.uint8)
            acc = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
        elif MODE == 6:
            sb = tl.load(sb_ptr + rn[:, None] * KS + (k // 32 + rs)[None, :])
            acc = tl.dot_scaled(a, None, "e4m3", b, sb, "e4m3", acc)
        else:  # MODE == 7
            sa = tl.full((BM, BK // 32), 127, tl.uint8)
            sb = tl.load(sb_ptr + rn[:, None] * KS + (k // 32 + rs)[None, :])
            acc = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


CONFIGS = [  # BM, BN, BK, num_warps, num_stages — same list for every mode
    (128, 128, 128, 4, 3),
    (128, 128, 128, 8, 3),
    (128, 256, 128, 8, 3),
    (128, 128, 64, 4, 4),
]
SHAPES = [(4096, 4096, 4096), (8192, 8192, 8192), (2048, 8192, 8192)]
REPEATS = 3


def mma_kinds(ptx):
    kinds = sorted(set(re.findall(r"tcgen05\.mma[\w.:]*?kind::(\w+)", ptx)))
    if not kinds and "tcgen05.mma" not in ptx:
        if "wgmma" in ptx:
            return "wgmma"
        if "mma.sync" in ptx:
            return "mma.sync"
        return "no-mma"
    return ",".join(kinds) if kinds else "tcgen05(unknown kind)"


_E8M0_TABLE = {}


def scale_factor(S):
    """E8M0 uint8 [R, K//32] -> fp32 multiplier [R, K].

    Uses a 256-entry lookup table instead of torch.exp2: on B300 (sm_103) torch 2.8+cu128 JIT-compiles exp2 via
    NVRTC 12.8, which rejects -arch sm_103. Indexing is a precompiled kernel. Powers of two are exact either way.
    """
    dev = S.device
    if dev not in _E8M0_TABLE:
        _E8M0_TABLE[dev] = torch.tensor([2.0 ** (i - 127) for i in range(256)], dtype=torch.float32, device=dev)
    return _E8M0_TABLE[dev][S.long()].repeat_interleave(32, dim=1)


def reference(A, Bt, SA, SB, mode):
    use_a, use_b = USES[mode]
    a = A.float() * (scale_factor(SA) if use_a else 1.0)
    b = Bt.float() * (scale_factor(SB) if use_b else 1.0)
    return a @ b.T


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}>"


def triton_checkout():
    """Directory of the git checkout the imported triton comes from (editable install), or None."""
    d = os.path.dirname(os.path.abspath(triton.__file__))
    for _ in range(4):
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        d = os.path.dirname(d)
    return None


def cuobjdump_path():
    p = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "cuobjdump")
    return p if os.path.exists(p) else "cuobjdump"


def environment(log):
    tdir = triton_checkout()
    commit = run(["git", "-C", tdir, "rev-parse", "HEAD"]) if tdir else "<not a git checkout>"
    dirty = run(["git", "-C", tdir, "status", "--porcelain", "--untracked-files=no"]) if tdir else ""
    log(f"device {torch.cuda.get_device_name()} cap {torch.cuda.get_device_capability()}")
    log(f"torch {torch.__version__} cuda {torch.version.cuda}")
    log(f"triton {triton.__version__} at {triton.__file__}")
    log(f"triton checkout {tdir} commit {commit} dirty={'yes' if dirty else 'no'}")
    log(f"nvidia-smi {run(['nvidia-smi', '--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem', '--format=csv,noheader'])}")
    ptxas = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas")
    log(f"ptxas {ptxas}: {run([ptxas, '--version']).splitlines()[-1:] if os.path.exists(ptxas) else '<missing>'}")
    log(f"cuobjdump {cuobjdump_path()}")
    return commit, bool(dirty)


def save_artifacts(art, tag, k):
    """Write TTGIR, PTX, SASS for one compiled kernel; return SASS instruction counts."""
    os.makedirs(art, exist_ok=True)
    base = os.path.join(art, tag)
    for key, ext in (("ttgir", "ttgir"), ("ptx", "ptx"), ("llir", "ll")):
        if key in k.asm:
            with open(f"{base}.{ext}", "w") as f:
                f.write(k.asm[key])
    counts = {}
    if "cubin" in k.asm:
        cub = f"{base}.cubin"
        with open(cub, "wb") as f:
            f.write(k.asm["cubin"])
        sass = run([cuobjdump_path(), "-sass", cub])
        with open(f"{base}.sass", "w") as f:
            f.write(sass)
        ops = collections.Counter(m.group(1) for m in re.finditer(r"^\s+/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9]*)", sass, re.M))
        # Opcode base names only: "F2FP.BF16.F32.PACK_AB" counts as F2FP.
        counts = {op: ops.get(op, 0) for op in SASS_OPS}
    return counts


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default=None, help="comma list; default all (or QUICK_MODES with --quick)")
    ap.add_argument("--shapes", default=None, help="comma list of M values; default all (or 8192 with --quick)")
    ap.add_argument("--quick", action="store_true", help="8192^3 only, the one-missing-scale modes + their workarounds")
    ap.add_argument("--out", default="o1_fresh_results.txt")
    ap.add_argument("--artifacts", default=None, help="directory for TTGIR/PTX/SASS of the best config per mode")
    ap.add_argument("--expect-commit", default=None, help="fail (exit 1) if the imported triton is not this commit")
    ap.add_argument("--no-random-check", action="store_true")
    args = ap.parse_args()
    mode_names = (args.modes.split(",") if args.modes else (QUICK_MODES if args.quick else list(MODES)))
    shape_sel = (args.shapes.split(",") if args.shapes else (["8192"] if args.quick else None))
    modes = {m: MODES[m] for m in mode_names}
    shapes = SHAPES if shape_sel is None else [s for s in SHAPES if str(s[0]) in shape_sel]
    out = open(args.out, "w")
    failures, unexpected = [], []

    def log(*a):
        print(*a)
        print(*a, file=out)
        out.flush()

    commit, dirty = environment(log)
    if args.expect_commit and not commit.startswith(args.expect_commit[:12]):
        failures.append(f"imported triton commit {commit} != expected {args.expect_commit}")
        log(f"FAIL: {failures[-1]}")
    if dirty:
        log("WARNING: the triton checkout has uncommitted changes")
    log(f"modes {list(modes)} shapes {shapes} repeats {REPEATS}")
    torch.manual_seed(0)
    for si, (M, N, K) in enumerate(shapes):
        A = torch.randint(-3, 4, (M, K), device="cuda").to(torch.float8_e4m3fn)
        Bt = torch.randint(-3, 4, (N, K), device="cuda").to(torch.float8_e4m3fn)
        bufs = {
            "unit": (torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda"),
                     torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")),
            "var": (torch.randint(124, 131, (M, K // 32), device="cuda").to(torch.uint8),
                    torch.randint(124, 131, (N, K // 32), device="cuda").to(torch.uint8)),
        }
        flops = 2.0 * M * N * K
        log(f"\n=== M={M} N={N} K={K}")
        for name, (mode, which) in modes.items():
            SA, SB = bufs[which]
            ref = reference(A, Bt, SA, SB, mode)
            best = None
            for (BM, BN, BK, nw, ns) in CONFIGS:
                C = torch.empty((M, N), dtype=torch.float32, device="cuda")
                grid = (M // BM, N // BN)

                def launch():
                    return mm_kernel[grid](A, Bt, SA, SB, C, M, N, K, BM=BM, BN=BN, BK=BK, MODE=mode,
                                           num_warps=nw, num_stages=ns)
                try:
                    k = launch()
                    torch.cuda.synchronize()
                except Exception as e:  # noqa: BLE001
                    log(f"  {name:12s} cfg={BM}x{BN}x{BK} w{nw} s{ns}: COMPILE/RUN ERROR {type(e).__name__}: {str(e)[:160]}")
                    continue
                exact = torch.equal(C, ref)
                if not exact:
                    failures.append(f"{name} M={M} cfg={BM}x{BN}x{BK} w{nw} s{ns}: not exact")
                kinds = mma_kinds(k.asm["ptx"])
                regs = getattr(k, "n_regs", "?")
                spills = getattr(k, "n_spills", "?")
                tfs = [flops / triton.testing.do_bench(launch, warmup=25, rep=100) / 1e9 for _ in range(REPEATS)]
                med = statistics.median(tfs)
                log(f"  {name:12s} cfg={BM}x{BN}x{BK} w{nw} s{ns}: {med:8.1f} TFLOP/s (min {min(tfs):.1f} max {max(tfs):.1f})"
                    f"  exact={exact}  mma={kinds}  regs={regs} spills={spills}")
                if exact and (best is None or med > best[0]):
                    best = (med, (BM, BN, BK, nw, ns), kinds, k)
            if not best:
                log(f"  BEST {name:12s}: none (no exact run)")
                failures.append(f"{name} M={M}: no exact run")
                continue
            med, cfg, kinds, kbest = best
            exp = EXPECTED_KIND[name]
            flag = "" if kinds == exp else f"  UNEXPECTED (expected {exp})"
            if flag:
                unexpected.append(f"{name} M={M}: mma={kinds}, expected {exp}")
            log(f"  BEST {name:12s}: {med:8.1f} TFLOP/s cfg={cfg} mma={kinds}{flag}")
            if args.artifacts:
                tag = f"M{M}_{name.replace('+', '_plus_')}_{'x'.join(map(str, cfg))}"
                counts = save_artifacts(args.artifacts, tag, kbest)
                log(f"  SASS {name:12s}: " + " ".join(f"{op}={n}" for op, n in counts.items()))
            if si == 0 and not args.no_random_check and mode != 3:
                # Random e4m3 values and random varying scales, with the best config: catches anything the
                # small-integer inputs could hide. fp32 summation order differs, so allclose, not equal.
                BM, BN, BK, nw, ns = cfg
                Ar = torch.randn((M, K), device="cuda").to(torch.float8_e4m3fn)
                Btr = torch.randn((N, K), device="cuda").to(torch.float8_e4m3fn)
                SAr = torch.randint(120, 135, (M, K // 32), device="cuda").to(torch.uint8)
                SBr = torch.randint(120, 135, (N, K // 32), device="cuda").to(torch.uint8)
                Cr = torch.empty((M, N), dtype=torch.float32, device="cuda")
                mm_kernel[(M // BM, N // BN)](Ar, Btr, SAr, SBr, Cr, M, N, K, BM=BM, BN=BN, BK=BK, MODE=mode,
                                               num_warps=nw, num_stages=ns)
                torch.cuda.synchronize()
                refr = reference(Ar, Btr, SAr, SBr, mode)
                err = (Cr - refr).abs()
                rel = (err.max() / refr.abs().max()).item()
                ok = torch.allclose(Cr, refr, rtol=1e-3, atol=1e-3 * refr.abs().max().item())
                if not ok:
                    failures.append(f"{name} M={M}: random allclose failed (max_rel_to_max={rel:.3e})")
                log(f"  RANDOM {name:12s}: allclose={ok} max_abs={err.max().item():.3e} max_rel_to_max={rel:.3e}")
        try:
            one = torch.tensor(1.0, device="cuda")
            ref = A.float() @ Bt.float().T
            f = lambda: torch._scaled_mm(A, Bt.T, scale_a=one, scale_b=one, out_dtype=torch.float32)  # noqa: E731
            exact = torch.equal(f(), ref)
            ms = triton.testing.do_bench(f, warmup=25, rep=100)
            log(f"  cuBLAS fp8 (_scaled_mm, per-tensor scale; reference only): {flops / ms / 1e9:8.1f} TFLOP/s  exact={exact}")
        except Exception as e:  # noqa: BLE001
            log(f"  cuBLAS fp8 (_scaled_mm): ERROR {type(e).__name__}: {str(e)[:160]}")
    log("\nSUMMARY")
    for f_ in failures:
        log(f"  FAIL: {f_}")
    for u in unexpected:
        log(f"  UNEXPECTED: {u}")
    status = 1 if failures else (2 if unexpected else 0)
    log(f"EXIT {status} ({'correctness/commit failure' if status == 1 else 'unexpected mma kind' if status == 2 else 'all checks passed'})")
    log("DONE")
    return status


if __name__ == "__main__":
    sys.exit(main())
