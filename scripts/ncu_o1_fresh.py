#!/usr/bin/env python3
"""Run ONE O1 variant for Nsight Compute: 3 warm-up launches, then 1 launch to profile.

    ncu --kernel-name mm_kernel --launch-skip 3 --launch-count 1 <sections> -o out \
        python3 ncu_o1_fresh.py MODE BMxBNxBK:WARPS:STAGES

MODE: any key of bench_o1_fresh.MODES (e.g. lhsvar, lhsvar+const, rhsvar, rhsvar+const).
Shape 8192^3, same inputs/scales as bench_o1_fresh.py (seed 0). Prints whether the result is exact.
"""
import sys

import torch

from bench_o1_fresh import MODES, mm_kernel, reference

name, cfg = sys.argv[1], sys.argv[2]
tile, nw, ns = cfg.split(":")
BM, BN, BK = (int(x) for x in tile.split("x"))
nw, ns = int(nw), int(ns)
mode, which = MODES[name]
M = N = K = 8192
torch.manual_seed(0)
A = torch.randint(-3, 4, (M, K), device="cuda").to(torch.float8_e4m3fn)
Bt = torch.randint(-3, 4, (N, K), device="cuda").to(torch.float8_e4m3fn)
if which == "unit":
    SA = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda")
    SB = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")
else:
    SA = torch.randint(124, 131, (M, K // 32), device="cuda").to(torch.uint8)
    SB = torch.randint(124, 131, (N, K // 32), device="cuda").to(torch.uint8)
C = torch.empty((M, N), dtype=torch.float32, device="cuda")
for _ in range(4):
    mm_kernel[(M // BM, N // BN)](A, Bt, SA, SB, C, M, N, K, BM=BM, BN=BN, BK=BK, MODE=mode,
                                   num_warps=nw, num_stages=ns)
torch.cuda.synchronize()
print(name, cfg, "exact", torch.equal(C, reference(A, Bt, SA, SB, mode)))
