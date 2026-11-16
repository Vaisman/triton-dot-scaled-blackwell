import re

import torch
import triton
import triton.language as tl


@triton.jit
def mm(a_ptr, bt_ptr, sa_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
       UNIT_RHS_SCALE: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    rs = tl.arange(0, BK // 32)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :])
        b = tl.load(bt_ptr + rn[None, :] * K + (k + rk)[:, None])
        sa = tl.load(sa_ptr + rm[:, None] * (K // 32) + (k // 32 + rs)[None, :])
        if UNIT_RHS_SCALE:
            sb = tl.full((BN, BK // 32), 127, tl.uint8)
            acc = tl.dot_scaled(a, sa, "e4m3", b, sb, "e4m3", acc)
        else:
            acc = tl.dot_scaled(a, sa, "e4m3", b, None, "e4m3", acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


M = N = K = 8192
torch.manual_seed(0)
A = torch.randint(-3, 4, (M, K), device="cuda").to(torch.float8_e4m3fn)
Bt = torch.randint(-3, 4, (N, K), device="cuda").to(torch.float8_e4m3fn)
SA = torch.randint(124, 131, (M, K // 32), device="cuda").to(torch.uint8)
pow2 = torch.tensor([2.0 ** (i - 127) for i in range(256)], device="cuda")
ref = (A.float() * pow2[SA.long()].repeat_interleave(32, dim=1)) @ Bt.float().T
for unit in (False, True):
    C = torch.empty((M, N), dtype=torch.float32, device="cuda")
    run = lambda: mm[(M // 128, N // 128)](A, Bt, SA, C, M, N, K, BM=128, BN=128, BK=128,
                                            UNIT_RHS_SCALE=unit, num_warps=4, num_stages=3)
    k = run()
    kind = sorted(set(re.findall(r"kind::(\w+)", k.asm["ptx"])))
    ms = triton.testing.do_bench(run)
    print(f"UNIT_RHS_SCALE={unit}: kind={kind} exact={torch.equal(C, ref)} {2 * M * N * K / ms / 1e9:.0f} TFLOP/s")
