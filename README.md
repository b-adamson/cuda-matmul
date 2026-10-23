# cuda-matmul

Single-precision square matrix multiplication (`C = A * B`, row-major) on a
Tesla T4, with three hand-written CUDA kernels benchmarked against cuBLAS and
profiled with Nsight Compute to explain where each one is limited.

| kernel | what it does |
|---|---|
| `naive` | one thread per output element, operands from global memory |
| `tiled` | 16x16 shared-memory tiles, one output element per thread |
| `regtiled` | 64x64x8 shared-memory tiles, 4x4 outputs per thread held in registers |
| `cublas` | `cublasSgemm`, pedantic FP32 math (no Tensor Cores) |

## Results

Tesla T4 (Google Colab), CUDA 13.0, median of 5 samples of 10 launches.
TFLOP/s counts `2*N^3` FLOPs.

| N | naive | tiled | regtiled | cuBLAS |
|---|---:|---:|---:|---:|
| 512  | 0.57 | 0.89 | 1.98 | 5.00 |
| 1024 | 0.50 | 0.81 | 2.26 | 5.62 |
| 2048 | 0.44 | 0.69 | 2.17 | 4.33 |
| 4096 | 0.44 | 0.67 | 2.05 | 4.23 |

At N=4096, `regtiled` reaches 48% of cuBLAS (`tiled` 16%, `naive` 10%).
All values come from one run in which the GPU was power-throttled (mean SM clock
1110 MHz against a 1590 MHz boost, at 68.6 W of a 70 W cap), so the nominal
8.1 TFLOP/s peak is not attainable; the FP32 limit at the observed clock is
about 5.7 TFLOP/s. Treat ratios between kernels as more reliable than absolute
rates, and see [Limitations](#limitations).

## Findings

1. **Shared-memory tiling alone gives 1.5-1.6x over naive.** It issues 1/16 as
   many global loads, but both kernels are limited by the L1/TEX unit (91% and
   95% busy), not the FMA pipes. For `tiled` the profile also rules out DRAM
   (72 GB/s of 320), L2 (15%) and occupancy (98.7%). Naive saturates L1/TEX through the global-load queue, tiled
   through the shared-memory path (49.7% of tiled's warp stall samples are MIO
   throttle).
2. **Register tiling gives 2.2-3.2x over `tiled`** (2.8-3.2x for N of 1024 and up). Shared-load instructions per
   FMA fall from 1.25 to 0.32, L1/TEX utilisation drops to 66%, and the largest
   stall becomes waiting on the global loads that fill the tiles (long
   scoreboard, 48% of the sampled reasons).
3. **cuBLAS is 2.1-2.5x faster than `regtiled`** and its throughput falls as N
   grows, consistent with the clock throttling above.

Details and data: [docs/profile-analysis.md](docs/profile-analysis.md),
[results/t4-run-001/](results/t4-run-001/).

## Reproduce

On a Colab T4 runtime (check `nvidia-smi` shows a Tesla T4):

    git clone https://github.com/b-adamson/cuda-matmul.git
    cd cuda-matmul
    make ARCH=-arch=sm_75
    ./matmul 65 all 1 1                       # quick correctness check on awkward sizes
    python3 scripts/baseline.py --ncu --output /content/drive/MyDrive/cuda-matmul/run-001

The runner validates every kernel against cuBLAS and a double-precision CPU
oracle at 13 small sizes, then times N=512-4096 and profiles `tiled`. To profile
the other kernels:

    ncu --set full --clock-control none --cache-control all --launch-count 1 \
        --export regtiled ./matmul --profile regtiled 1024

Full protocol, artifact list and validation details: [docs/T4.md](docs/T4.md).
Other GPUs: `make ARCH=-arch=sm_XX`. Python tests (no GPU needed):
`python3 -m unittest discover -s tests`.

## Limitations

- One timing run on one power-throttled Colab T4. Kernels run in a fixed order,
  so clock drift can bias the ratios; cuBLAS at N=2048 varied by 50% between
  samples. Repeat runs have not been done.
- One profile launch per kernel at N=1024, with unfixed clocks. The saturated
  sub-unit inside L1/TEX has not been identified, the vectorised-load inference
  has not been checked in the SASS, and the stall shares for `naive` and
  `regtiled` cover only seven extracted reasons.
- Results above N=512 are checked against cuBLAS in full but against the CPU
  oracle only at 32 sampled outputs.
- Tile sizes were chosen once and not tuned. Vectorised loads, double buffering
  and Tensor Cores have not been tried.

## Layout

| path | purpose |
|---|---|
| `matmul.cu` | the three kernels |
| `baseline.cuh` | harness: validation, event timing, `--profile` mode |
| `scripts/baseline.py` | runs the harness and records results, telemetry and metadata |
| `tests/` | tests for the runner's validation logic |
| `docs/T4.md` | measurement and profiling protocol |
| `docs/profile-analysis.md` | interpretation of the Nsight Compute profiles |
| `results/t4-run-001/` | timing summary and extracted profile metrics |
