import torch
import tk_kernel
import random
import time
import math
from torch.nn.functional import scaled_dot_product_attention
import aiter

torch.set_printoptions(
    precision=3,        # decimals
    sci_mode=False,     # True → always scientific
    linewidth=220,      # characters per line before folding
    threshold=float("inf")  # print every element, no summarising "..."
)

# Inputs
B = 16
H = 64
H_KV = 8
N = 8192
D = 128
causal = False
dtype = torch.bfloat16

def flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    """Calculate FLOPs for attention operation."""
    flop = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return flop

def efficiency(flop, time):
    """Calculate efficiency in TFLOPS."""
    flop = flop / 1e12  # convert to TFLOPS
    time = time / 1e3   # convert to seconds
    return flop / time


num_warmup = 20
num_iters = 20

start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = flops(B, N, H, D, causal)

# Reference matmul using AITER
for _ in range(num_warmup):
    q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    out_ref = aiter.flash_attn_func(q, k, v, causal=causal, return_lse=True, deterministic=True)
timings_ref = []
torch.manual_seed(0)
random.seed(0)
for _ in range(num_iters):
    q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    torch.cuda.synchronize()
    start_event.record()
    out_ref = aiter.flash_attn_func(q, k, v, causal=causal, return_lse=True, deterministic=True)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_ref.append(elapsed_time)
print(f"{out_ref[0].dtype=}")
avg_time_ref = sum(timings_ref) / len(timings_ref)
eff_ref = efficiency(flops_ref, avg_time_ref)
print(f"AITER (AMD) reference average execution time: {avg_time_ref:.4f} ms")
print(f"AITER (AMD) reference performance: {eff_ref:.2f} TFLOPS for {B=} {H=} {N=} {D=} {causal=}.\n")

# Kernel matmul
for _ in range(num_warmup):
    out = torch.zeros(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
    q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    tk_kernel.dispatch_micro(q, k, v, out)
timings = []
out = torch.zeros(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
torch.manual_seed(0)
random.seed(0)
for _ in range(num_iters):
    q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    torch.cuda.synchronize()
    start_event.record()
    tk_kernel.dispatch_micro(q, k, v, out)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
print(f"{out.dtype=}")
avg_time = sum(timings) / len(timings)
eff = efficiency(flops_ref, avg_time)
print(f"Average execution time: {avg_time:.4f} ms")
print(f"Performance: {eff:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")

# Compare against reference
out_float = out.float()
out_ref_float = out_ref[0].float()
diff = (out_float - out_ref_float)
max_error = diff.max().item()
mean_error = diff.mean().item()
error_count = (diff > 0.1).sum().item()
print(f"Max error between kernel and reference: {max_error}")
print(f"Max error: {max_error}")
print(f"Mean error: {mean_error}")
print(f"Number of large errors (>0.1): {error_count}\n")
print(out_float[0:2, 0, 0, :16])
print(out_ref_float[0:2, 0, 0, :16])

