

### Blackwell setup

We compare to CUTLASS, Python, and CUBLAS/CUDNN.

## BF16

CuBLAS:
```bash
cd AMD-benchmarking-harness/analysis/blackwell/bf16/
make clean && make
./matmul
```

CUTLASS:
```bash
# obtain library
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass 
mkdir build && cd build

cmake .. -DCUTLASS_NVCC_ARCHS=100a -DCUTLASS_UNITY_BUILD_ENABLED=OFF -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_ENABLE_EXAMPLES=OFF -DCUTLASS_LIBRARY_KERNELS=cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_void_bf16*tnt*
make cutlass_profiler -j64

./tools/profiler/cutlass_profiler --operation=tensorop_gemm --m=1024 --n=1024 --k=1024 --output=bf16.csv --kernels="cutlass3x_sm100_tensorop_gemm_bf16_bf16" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1

./tools/profiler/cutlass_profiler --operation=tensorop_gemm --m=2048 --n=2048 --k=2048 --output=bf16.csv --kernels="cutlass3x_sm100_tensorop_gemm_bf16_bf16" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1

./tools/profiler/cutlass_profiler --operation=tensorop_gemm --m=4096 --n=4096 --k=4096 --output=bf16.csv --kernels="cutlass3x_sm100_tensorop_gemm_bf16_bf16" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1

./tools/profiler/cutlass_profiler --operation=tensorop_gemm --m=8192 --n=8192 --k=8192 --output=bf16.csv --kernels="cutlass3x_sm100_tensorop_gemm_bf16_bf16" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1

./tools/profiler/cutlass_profiler --operation=tensorop_gemm --m=16384 --n=16384 --k=16384 --output=bf16.csv --kernels="cutlass3x_sm100_tensorop_gemm_bf16_bf16" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1
```


## FP8

CuBLAS:
```bash
cd AMD-benchmarking-harness/analysis/blackwell/fp8/
make clean && make
./matmul
```

```bash
cmake .. -DCUTLASS_NVCC_ARCHS=100a -DCUTLASS_UNITY_BUILD_ENABLED=OFF -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_ENABLE_EXAMPLES=OFF -DCUTLASS_LIBRARY_KERNELS=cutlass3x_sm100_bstensorop_gemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_bf16*tnt*
make cutlass_profiler -j64

./tools/profiler/cutlass_profiler --operation=bstensorop_gemm --m=1024 --n=1024 --k=1024 --output=mfxe4m3.csv --kernels="cutlass3x_sm100_bstensorop_gemm_ue8m0xe4m3_ue8m0xe4m3" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1

./tools/profiler/cutlass_profiler --operation=bstensorop_gemm --m=2048 --n=2048 --k=2048 --output=mfxe4m3.csv --kernels="cutlass3x_sm100_bstensorop_gemm_ue8m0xe4m3_ue8m0xe4m3" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1

./tools/profiler/cutlass_profiler --operation=bstensorop_gemm --m=4096 --n=4096 --k=4096 --output=mfxe4m3.csv --kernels="cutlass3x_sm100_bstensorop_gemm_ue8m0xe4m3_ue8m0xe4m3" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1

./tools/profiler/cutlass_profiler --operation=bstensorop_gemm --m=8192 --n=8192 --k=8192 --output=mfxe4m3.csv --kernels="cutlass3x_sm100_bstensorop_gemm_ue8m0xe4m3_ue8m0xe4m3" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1

./tools/profiler/cutlass_profiler --operation=bstensorop_gemm --m=16384 --n=16384 --k=16384 --output=mfxe4m3.csv --kernels="cutlass3x_sm100_bstensorop_gemm_ue8m0xe4m3_ue8m0xe4m3" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1
```


## FP6 

```bash
# obtain library
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass 
mkdir build && cd build

# isolate desired kernels and build
cmake .. -DCUTLASS_NVCC_ARCHS=100a -DCUTLASS_UNITY_BUILD_ENABLED=OFF -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_ENABLE_EXAMPLES=OFF -DCUTLASS_LIBRARY_KERNELS=cutlass3x_sm100_bstensorop_gemm_ue8m0xe2m3_ue8m0xe2m3_f32_void_bf16*tnt*
make cutlass_profiler -j64

# test
./tools/profiler/cutlass_profiler --operation=bstensorop_gemm --m=1024 --n=1024 --k=1024 --output=mfxe2m3.csv --kernels="cutlass3x_sm100_bstensorop_gemm_ue8m0xe2m3_ue8m0xe2m3" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1

./tools/profiler/cutlass_profiler --operation=bstensorop_gemm --m=2048 --n=2048 --k=2048 --output=mfxe2m3.csv --kernels="cutlass3x_sm100_bstensorop_gemm_ue8m0xe2m3_ue8m0xe2m3" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1

./tools/profiler/cutlass_profiler --operation=bstensorop_gemm --m=4096 --n=4096 --k=4096 --output=mfxe2m3.csv --kernels="cutlass3x_sm100_bstensorop_gemm_ue8m0xe2m3_ue8m0xe2m3" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1

./tools/profiler/cutlass_profiler --operation=bstensorop_gemm --m=8192 --n=8192 --k=8192 --output=mfxe2m3.csv --kernels="cutlass3x_sm100_bstensorop_gemm_ue8m0xe2m3_ue8m0xe2m3" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1

./tools/profiler/cutlass_profiler --operation=bstensorop_gemm --m=16384 --n=16384 --k=16384 --output=mfxe2m3.csv --kernels="cutlass3x_sm100_bstensorop_gemm_ue8m0xe2m3_ue8m0xe2m3" --profiling-duration=2 --min-iterations=20 --verification-enabled=false --dist=uniform,min:-1,max:1
```

## Utils.

For each set of outputs from CUTLASS, you can scrape for the best result using something like:

```python
import pandas as pd
import glob

# Load all relevant profiler CSVs
files = glob.glob("mfxe2m3*.csv")

dfs = []
for f in files:
    try:
        df = pd.read_csv(f)
        df["source"] = f
        dfs.append(df)
    except Exception as e:
        print(f"skip {f}: {e}")

if not dfs:
    raise SystemExit("no CSVs found")

df = pd.concat(dfs, ignore_index=True)

# Filter only successful GEMMs
df = df[df["Status"].str.contains("success", na=False)]
df = df[df["OperationKind"].str.contains("gemm", na=False)]

# Find the row with max GFLOPs
best = df.loc[df["GFLOPs"].idxmax()]

print("\n=== Best kernel ===")
print(f"File:       {best['source']}")
print(f"Operation:  {best['Operation']}")
print(f"GFLOPs:     {best['GFLOPs']:.2f}")
print(f"Runtime:    {best['Runtime']:.6f} s")
print(f"GB/s:       {best['GB/s']:.2f}")
print(f"Disposition:{best['Disposition']}")
print(f"m,n,k:      {best['m']},{best['n']},{best['k']}")

# Optional: save top-10 summary
top10 = df.sort_values("GFLOPs", ascending=False).head(10)
top10[["Operation","Runtime","GB/s","GFLOPs","source"]].to_csv("cutlass_top10.csv", index=False)
print("\nWrote top10 → cutlass_top10.csv")
```
