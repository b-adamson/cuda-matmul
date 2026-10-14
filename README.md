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
| Peak FP32 | _fill in_ |
| Peak bandwidth | _fill in_ |

## Results

| kernel | N | ms | GFLOP/s | % of peak FLOPs | GB/s | % of peak BW |
|---|---|---|---|---|---|---|
| naive | 1024 | | | | | |
| tiled | 1024 | | | | | |

## What I expected vs what happened

_Write this before profiling, then check it with Nsight Compute later._

## Profiling (later)

Nsight Compute needs GPU performance counters, which containerised providers
(RunPod, Colab) do not grant — `ERR_NVGPUCTRPERM`. Needs a VM or bare metal
where you are root on the host, or a department machine.

    ncu --set full ./matmul 1024
