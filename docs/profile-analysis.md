# Profile analysis (Tesla T4, N=1024)

One launch of each custom kernel under Nsight Compute (`--set full
--clock-control none --cache-control all`). Clocks were not fixed and the T4 was
power-throttled, so use the proportions, not the absolute times: profiler
durations differ from the benchmark run (naive 3.51 ms vs 4.31 ms, tiled 2.34 vs
2.67, regtiled 1.31 vs 0.95). Data: `results/t4-run-001/profile-metrics.csv`.

## Summary

| | naive | tiled | regtiled |
|---|---:|---:|---:|
| L1/TEX throughput, % of peak | 91.1 | 94.5 | 66.0 |
| Instruction throughput, % of peak | 20.3 | 26.7 | 35.5 |
| Registers per thread | 49 | 39 | 72 |
| Shared-load instructions per FMA (per warp) | n/a | 1.25 | 0.32 |
| Largest stall reason (of the 7 reasons extracted) | LG throttle, 89% | MIO throttle, 58% | long scoreboard, 48% |

Shared-load instructions per FMA are computed from counters: tiled
41,943,040 / (32,768 warps x 64 phases x 16 FMAs); regtiled
10,747,904 / (2,048 warps x 128 phases x 128 FMAs).

Stall shares among the seven reasons extracted for all three kernels
(barrier, LG throttle, long scoreboard, math-pipe throttle, MIO throttle,
selected, short scoreboard):

| stall | naive | tiled | regtiled |
|---|---:|---:|---:|
| LG throttle (global/local load queue full) | 89.3% | 0.1% | 0.2% |
| MIO throttle (shared/special-function queue full) | 0.0% | 57.8% | 29.7% |
| Long scoreboard (waiting on global load data) | 7.6% | 20.7% | 48.4% |
| Barrier | 0.0% | 15.4% | 7.8% |
| Selected (issuing an instruction) | 2.6% | 3.5% | 9.0% |
| Math-pipe throttle | 0.3% | 0.8% | 3.3% |

## What this shows

**Naive and tiled are both limited by the L1/TEX unit.** It is 91% and 95% busy,
while the FMA pipes are barely contended (math-pipe throttle at most 0.8% of
the sampled stalls), instruction issue is at 20-27% of peak, and, for tiled,
DRAM throughput was 72 GB/s (about 23% of 320 GB/s), L2 throughput 15%, and
achieved occupancy 98.7%. The two kernels saturate it through different queues:
naive through the global-load queue (LG throttle), tiled through the shared
memory path (MIO throttle). That is consistent with tiling giving only
1.5-1.6x over naive despite issuing 1/16 as many global loads.

For tiled, shared-memory accesses have zero bank conflicts and wavefronts match
the ideal count, and the shared data pipe alone is 40% of peak, so the 94.5%
L1/TEX figure is set by another part of the unit (probably request or
instruction side). That sub-unit has not been identified.

**Register tiling moves the bottleneck.** L1/TEX drops to 66%, shared-load
instructions per FMA fall from 1.25 to 0.32, instruction throughput rises to
35%, and the issuing share of warp samples rises from 3.5% to 9.0%. The largest
stall is now long scoreboard (waiting for the global loads that fill the
shared tiles), not MIO throttle. The math pipe is still not saturated (3.3%
throttle; 2.05 TFLOP/s at N=4096 against a clock-limited FP32 limit of about
5.7 TFLOP/s). The regtiled kernel uses 72 registers per thread; its occupancy and
spill behaviour were not extracted.

## Limits of this analysis

- One profile launch per kernel, under unfixed power-throttled clocks.
- PC-sampling stall counts are sample counts. Shares above are shares of the
  seven reasons extracted, not of all stalls (for tiled the full set is
  132,525 samples, in which MIO throttle is 49.7%).
- Shared-load counts per FMA assume 16 (tiled) and 128 (regtiled) FMAs per
  warp-phase; the tiled count of 20 loads per warp-phase suggests the compiler
  vectorised the `As` reads, which was inferred from the counts and not checked
  in the SASS.
- Not done: profiles at other sizes, L1/TEX sub-unit breakdown, SASS inspection,
  repeat runs to bound clock drift.
