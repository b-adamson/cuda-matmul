# T4 run 001

Tesla T4 on Google Colab, 23 October 2026. See `docs/T4.md` for the protocol and
`docs/profile-analysis.md` for the interpretation.

- `summary.csv`: median/min/max time and TFLOP/s per kernel and size
  (5 samples of 10 launches each, from `scripts/baseline.py`).
- `profile-metrics.csv`: selected Nsight Compute metrics for one launch of each
  custom kernel at N=1024 (`ncu --set full --clock-control none --cache-control all`).
  The seven stall reasons listed are only those extracted, not the full set, so
  stall shares computed from this file are shares of those seven reasons.

Not in the repository: raw samples, telemetry, `nvidia-smi` dumps and the
`.ncu-rep` reports, which are on the run's Google Drive folder. During the
timed run the SM clock averaged 1110 MHz (minimum 1050) at 68.6 W mean power
against the 70 W cap, over 41 one-second samples with utilisation above 50%.
