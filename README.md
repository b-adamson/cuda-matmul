# cuda-matmul

Naive vs shared-memory-tiled matrix multiply, measured. The point is not the
kernel — it is being able to say *why* one is faster, with numbers.

## Build

    make                 # edit ARCH in the Makefile for a different card
    ./matmul 512         # small: verifies against a CPU reference
    ./matmul 1024        # measurement size

## Method

1. Write the naive kernel. Verify at N=512 before believing any timing.
2. Derive, on paper, how many bytes it reads from global memory. Fill in
   `bytes_naive`. Do this *before* running it.
3. Measure. Compare achieved GFLOP/s and GB/s against the card's peak.
   Whichever is closer to peak is the bottleneck.
4. Write the tiled kernel. Derive its byte count. Measure again.
5. Explain the ratio between the two.

## Card

| | |
|---|---|
| GPU | RTX A6000 (sm_86) |
| Peak FP32 | ~38.7 TFLOP/s (from spec sheet, verify) |
| Peak bandwidth | 768 GB/s (from spec sheet, verify) |

## Results

| kernel | N | ms | GFLOP/s | % of peak FLOPs | GB/s (issued) | issued / peak BW |
|---|---|---|---|---|---|---|
| naive | 1024 | 0.912 | 2356 | ~6% | 9423 | ~12.3x |
| tiled | 1024 | 0.704 | 3050 | ~8% | 763 | ~1.0x |

GB/s is *issued* load bytes divided by run time (`bytes_naive` = 2N^3 * 4,
`bytes_tiled` = 2N^3 / TILE * 4), not bytes served by DRAM. The last column is
therefore not DRAM utilisation. The minimum DRAM traffic is each of A and B
read once, 2 * N^2 * 4 = 8.4 MB, which over 0.704 ms is about 12 GB/s, under 2%
of peak.

## What I expected vs what happened

_Write this before profiling, then check it with Nsight Compute later._

**Expected:** ~16x speedup, from tiling issuing 1/TILE as many global loads.

**What happened.** Tiled was 1.29x faster than naive, not the ~16x the load
count suggests (tiling issues 1/TILE as many global loads).

- Naive "issues" 9.4 TB/s, about 12x the card's DRAM bandwidth, so DRAM cannot
  be serving those loads. The L1/L2 caches are absorbing most of the repeated
  reads, which is why removing them gained so little.
- Tiled issues 763 GB/s, which coincidentally matches the 768 GB/s DRAM peak.
  That is not a sign it is DRAM-bound, since the real DRAM traffic is far lower.
- Both kernels reach only 6-8% of peak FP32, so neither is compute-bound, and
  DRAM bandwidth is not the limit either. Something else is the bottleneck.

**Hypotheses to check with ncu** (not yet verified):

- Shared-memory throughput: the tiled inner loop does two shared loads per FMA.
- Latency / low instruction-level parallelism: one accumulator per thread, so a
  serial dependency chain on `sum`.
- Occupancy: warps resident per SM.

Considered and judged unlikely: shared-memory *writes* (2 per phase per thread,
versus 32 shared *reads*) and the write to C (N^2 floats once, and the naive
kernel does it too).

## Profiling (later)

Nsight Compute needs GPU performance counters, which containerised providers
(RunPod, Colab) do not grant — `ERR_NVGPUCTRPERM`. Needs a VM or bare metal
where you are root on the host, or a department machine.

    ncu --set full ./matmul 1024
