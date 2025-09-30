import torch
import random
import math
import tk_kernel

torch.set_printoptions(
    precision=3,
    sci_mode=False,
    linewidth=220,
    threshold=float("inf")
)

random.seed(0)
torch.manual_seed(0)

num_elements = 2048

A = torch.randn((1, 1, 16, 32), dtype=torch.bfloat16, device='cuda')
B = torch.randn((1, 1, 16, 32), dtype=torch.bfloat16, device='cuda')
C = torch.zeros((1, 1, 16, 16), dtype=torch.float32, device='cuda')

# reference
C_ref = torch.matmul(A, B.transpose(-1, -2).contiguous())

# tk
C_tk = torch.zeros_like(C)
torch.cuda.synchronize()
tk_kernel.dispatch_micro(A, B, C_tk)
torch.cuda.synchronize()

# check
print(C_tk[0, 0, :16, :16])
print(C_ref[0, 0, :16, :16])

diff = (C_ref - C_tk).abs().max()
print(f"diff: {diff}")