"""Run from Colab/Linux after make; standard library only."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["kernel", "n", "sample", "iterations", "ms", "tflops", "correct"]


def capture(command, output, required=True):
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    except OSError as error:
        output.write_text(str(error) + "\n")
        if required:
            raise
        return ""
    output.write_text(result.stdout + result.stderr)
    if required and result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command}; see {output}")
    return result.stdout


def read_samples(text, n, samples, iterations):
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != FIELDS:
        raise ValueError("Unexpected benchmark CSV header")
    rows = list(reader)
    expected = {(kernel, str(sample)) for kernel in ("naive", "tiled", "regtiled", "cublas")
                for sample in range(samples)}
    observed = {(row["kernel"], row["sample"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("Missing or duplicate benchmark samples")
    for row in rows:
        if (int(row["n"]) != n or int(row["iterations"]) != iterations
                or row["correct"] != "true"):
            raise ValueError("Benchmark validation failed")
        ms, tflops = float(row["ms"]), float(row["tflops"])
        if not (0 < ms < float("inf") and 0 < tflops < float("inf")):
            raise ValueError("Invalid timing sample")
        if abs(tflops / (2 * n**3 / (ms * 1e9)) - 1) > 1e-5:
            raise ValueError("Inconsistent FLOP rate")
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["kernel"], row["n"]), []).append(float(row["ms"]))
    result = []
    for (kernel, n), values in groups.items():
        median = statistics.median(values)
        result.append(dict(kernel=kernel, n=n, samples=len(values),
                           median_ms=median, min_ms=min(values), max_ms=max(values),
                           median_tflops=2 * int(n)**3 / (median * 1e9)))
    return result


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, help="New directory, e.g. on mounted Google Drive")
    parser.add_argument("--ncu", action="store_true", help="Profile tiled after ordinary timing")
    parser.add_argument("--profile-size", type=int, default=1024)
    args = parser.parse_args()
    if min(args.sizes + [args.iterations, args.samples, args.profile_size]) < 1:
        parser.error("sizes, iterations and samples must be positive")
    if max(args.sizes + [args.profile_size]) > 32768:
        parser.error("maximum supported matrix size is 32768")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or ROOT / "results" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"started_utc": stamp, "platform": platform.platform(),
                "arguments": vars(args) | {"output": str(output)}, "status": "running",
                "commands": [], "sha256": {}}
    rows = []
    monitor = None
    with (output / "telemetry.csv").open("w") as telemetry:
        try:
            source = output / "source"
            source.mkdir()
            for name in ("matmul.cu", "baseline.cuh", "Makefile", "scripts/baseline.py", "docs/T4.md"):
                path = ROOT / name
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                metadata["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            binary = ROOT / "matmul"
            metadata["sha256"]["matmul"] = hashlib.sha256(binary.read_bytes()).hexdigest()
            def run(command, filename, required=True):
                metadata["commands"].append(command)
                return capture(command, output / filename, required)
            for command, filename in [(["git", "rev-parse", "HEAD"], "commit.txt"),
                                      (["git", "status", "--short"], "git-status.txt"),
                                      (["git", "diff", "--no-ext-diff", "HEAD"], "working-tree.patch"),
                                      (["nvcc", "--version"], "nvcc.txt"),
                                      (["ncu", "--version"], "ncu-version.txt"),
                                      (["nvidia-smi", "-q"], "gpu-before.txt")]:
                run(command, filename, required=False)
            # Best effort: retain an error in the telemetry file if unsupported.
            try:
                monitor = subprocess.Popen([
                    "nvidia-smi", "--query-gpu=timestamp,uuid,name,clocks.sm,clocks.mem,"
                    "temperature.gpu,power.draw,pstate,utilization.gpu", "--format=csv", "-l", "1"],
                    stdout=telemetry, stderr=telemetry)
            except OSError as error:
                telemetry.write(str(error) + "\n")
            executable = str(binary)
            # Exercise both sides of TILE=16, BK=8 and BM=BN=64 boundaries.
            for n in [1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129]:
                text = run([executable, str(n), "all", "1", "1"], f"check-{n}.log")
                read_samples(text, n, 1, 1)
            for n in args.sizes:
                print(f"Benchmarking N={n}", flush=True)
                text = run([executable, str(n), "all", str(args.iterations), str(args.samples)],
                           f"benchmark-{n}.log")
                rows.extend(read_samples(text, n, args.samples, args.iterations))
                write_csv(output / "samples.csv", rows, FIELDS)
            summary = summarize(rows)
            write_csv(output / "summary.csv", summary, list(summary[0]))
            if args.ncu:
                run(["ncu", "--set", "full", "--clock-control", "none", "--cache-control", "all",
                     "--launch-count", "1", "--export", str(output / "tiled"),
                     executable, "--profile", "tiled", str(args.profile_size)], "ncu.log")
            metadata["status"] = "complete"
        except Exception as error:
            metadata["status"] = "failed"
            metadata["error"] = str(error)
            raise
        finally:
            if monitor is not None:
                monitor.terminate()
                try:
                    monitor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    monitor.kill()
                    monitor.wait()
            capture(["nvidia-smi", "-q"], output / "gpu-after.txt", required=False)
            metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
            (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
