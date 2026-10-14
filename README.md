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
| Peak FP32 | 38.7 TFLOP/s (NVIDIA datasheet, at 1800 MHz boost) |
| Peak bandwidth | 768 GB/s (NVIDIA datasheet, 384-bit GDDR6) |

## Results

All on the RTX A6000. `% of peak` is against 38.7 TFLOP/s FP32.

| kernel | N | ms | GFLOP/s | % of peak | % of cuBLAS |
|---|---|---|---|---|---|
| naive  | 1024 |  0.912 |  2355 | 6.1% | 12.8% |
| tiled  | 1024 |  0.703 |  3053 | 7.9% | 16.6% |
| cuBLAS | 1024 |  0.117 | 18430 | 47.6% | 100% |
| naive  | 2048 |  7.659 |  2243 | 5.8% |  9.4% |
| tiled  | 2048 |  5.907 |  2909 | 7.5% | 12.2% |
| cuBLAS | 2048 |  0.719 | 23900 | 61.8% | 100% |
| naive  | 4096 | 62.771 |  2190 | 5.7% |  8.8% |
| tiled  | 4096 | 46.678 |  2944 | 7.6% | 11.8% |
| cuBLAS | 4096 |  5.516 | 24915 | 64.4% | 100% |

cuBLAS is `cublasSgemm` on the same matrices (row-major handled by swapping the
operands), timed identically and checked against the CPU reference at N=512. At
18-25 TFLOP/s it is below the 38.7 TFLOP/s FP32 peak, so tensor cores do not
appear to be involved and it is a like-for-like baseline. Results above N=512
are timings only; correctness is checked at N=512.

## What I predicted, and what actually happened

**Prediction 1 (wrong): tiling would give ~16x**, since it issues 1/TILE as many
global loads. It gave 1.29x.

**Prediction 2 (wrong): the naive kernel was surviving because the matrices fit
in L2.** Growing N past the 6 MB of L2 should then make it collapse. It did not:
naive went 2355 -> 2243 -> 2190 GFLOP/s across a 64x increase in data (each
matrix is 64 MB at N=4096).

### What the flatness means

A kernel limited by data volume should degrade as the data outgrows the caches.
Naive is flat across 64x of data, and the naive-to-tiled gap is only 1.3x
despite a 16x difference in issued global loads. Both point to a cost that is
fixed *per operation* and independent of problem size, not to data volume.

My working explanation is the ratio of **loads to FMAs** in the inner loop:

| kernel | FMAs | loads per FMA | loads served from |
|---|---|---|---|
| naive  | N^3 | 2 | global, mostly hitting L1/L2 |
| tiled  | N^3 | 2 | shared memory |
| cuBLAS | N^3 | well below 2 (not measured) | registers, presumably |

All three do the same N^3 FMAs. Tiling changed where the loads come from, not
how many there are per FMA, which is why a 16x reduction in global loads bought
only 29%. Shared memory is a cheaper place to load from; it is not fewer loads.

### A rough ceiling for tiled

Shared memory serves 32 banks x 4 bytes = 128 bytes per cycle per SM (to be
checked against NVIDIA's documentation before quoting). At 2 shared loads per
FMA that is 8 bytes per FMA, so 16 FMAs/cycle/SM = 32 FLOP/cycle/SM:

    84 SMs x 32 FLOP/cycle x 1.8 GHz = 4.8 TFLOP/s

Measured tiled is about 2.9 TFLOP/s, roughly 60% of that bound, with the rest
plausibly going to loop overhead, two `__syncthreads()` per phase and the serial
dependency on `sum`. The FP32 pipes could do 38.7 TFLOP/s, so the FMA units are
idle ~92% of the time waiting for operands. The arithmetic is not the
bottleneck; feeding it is.

### Why cuBLAS rises with N and mine do not

cuBLAS goes 18.4 -> 23.9 -> 24.9 TFLOP/s. A plausible reason is that at N=1024
it cannot fully saturate 84 SMs and launch and tail effects are a larger share
of a short run, so growing N amortises them away. Mine are flat because they
sit at a low ceiling from the start, and more work cannot lift it.

### The earlier "9.4 TB/s issued" figure was counted at the wrong level

In the naive kernel a warp's 32 threads read `A[row*N+k]`, the same address,
which is one broadcast and not 32 loads. `B[k*N+col]` is 32 consecutive floats,
which coalesce into a handful of transactions. `bytes_naive` is a thread-level
count; the memory system sees a small fraction of it. The ">12x DRAM bandwidth"
reading is therefore partly an artefact of counting at the wrong level, as well
as of cache hits.

No bank conflicts arise in the tiled kernel: `As[ty][k]` is the same address for
all threads sharing a `ty` (broadcast), and `Bs[k][tx]` spans 16 distinct banks.

The tiled kernel has no bounds guard and `N/TILE` truncates, so it is only
correct when N is a multiple of TILE. 512, 1024, 2048 and 4096 all are.

## Next: register tiling

Give each thread a TM x TN patch of output instead of one element. Per k-step it
loads TM values of A and TN of B into registers, then does the TM x TN outer
product. At 4x4 that is 8 loads for 16 FMAs, 0.5 loads per FMA, down from 2.
Identical arithmetic, a quarter of the loads.

**Prediction to test (written before building it):** the shared-bandwidth
ceiling moves from 4.8 to roughly 19 TFLOP/s. If the model is right, a 4x4
register-tiled kernel should land in the 10-19 TFLOP/s range, i.e. 40-75% of
cuBLAS, up from ~12%.

## Open question for the profiler

Flatness across N is also consistent with being limited by L2 bandwidth rather
than by load-issue rate. The 1.3x naive-to-tiled ratio argues against pure L2
bandwidth, but timings alone cannot fully separate the two. First thing to check
with `ncu`. Other hypotheses: shared-memory throughput, latency from the single
accumulator per thread, and occupancy.

## Profiling (later)

Nsight Compute needs GPU performance counters, which containerised providers
(RunPod, Colab) do not grant — `ERR_NVGPUCTRPERM`. The restriction is a driver
kernel-module setting on the host, so no container can lift it. Needs a VM or
bare metal with root on the host, or a department machine.

    ncu --set full ./matmul 1024
