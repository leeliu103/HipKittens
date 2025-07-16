import torch
import tk_kernel
import random

profiling = True
profiling_ref = False
torch.manual_seed(0)
random.seed(0)

# Inputs
N = 16
A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda') / 10.0  
C = torch.zeros(N, N, dtype=torch.bfloat16, device='cuda')
C_ref = torch.zeros(N, N, dtype=torch.bfloat16, device='cuda')  

tk_kernel.dispatch_micro(A, C, C_ref)

print("Out")
print(C[0:4, 0:8])
print("Ref")
print(C_ref[0:4, 0:8])

diff = C - C_ref
print(f"diff[0:4]")
print(diff[0:4])

print(f"diff[4:8]")
print(diff[4:8])

print(f"diff[8:12]")
print(diff[8:12])

print(f"diff[12:16]")
print(diff[12:16])


max_diff = diff.abs().max()
print(f"Max diff: {max_diff}")
