import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "baseline", Path(__file__).resolve().parents[1] / "scripts" / "baseline.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class SampleTests(unittest.TestCase):
    def csv_text(self):
        return "kernel,n,sample,iterations,ms,tflops,correct\n" + "".join(
            f"{name},100,0,10,1,0.002,true\n"
            for name in ("naive", "tiled", "regtiled", "cublas"))

    def test_complete_samples(self):
        self.assertEqual(len(baseline.read_samples(self.csv_text(), 100, 1, 10)), 4)

    def test_missing_sample(self):
        with self.assertRaises(ValueError):
            baseline.read_samples(self.csv_text(), 100, 2, 10)

    def test_duplicate_sample(self):
        with self.assertRaises(ValueError):
            baseline.read_samples(self.csv_text() + "naive,100,0,10,1,0.002,true\n", 100, 1, 10)

    def test_invalid_measurements(self):
        for bad in ("nan", "inf", "0", "-1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                baseline.read_samples(self.csv_text().replace(",1,0.002,", f",{bad},0.002,"), 100, 1, 10)

    def test_failed_correctness(self):
        with self.assertRaises(ValueError):
            baseline.read_samples(self.csv_text().replace("true", "false"), 100, 1, 10)

    def test_inconsistent_rate(self):
        with self.assertRaises(ValueError):
            baseline.read_samples(self.csv_text().replace("0.002", "2"), 100, 1, 10)

    def test_summary_uses_median_time(self):
        rows = [dict(kernel="tiled", n="100", ms=str(ms)) for ms in (1, 2, 90)]
        result = baseline.summarize(rows)[0]
        self.assertEqual(result["median_ms"], 2)
        self.assertEqual(result["median_tflops"], 0.001)
        self.assertEqual(result["samples"], 3)


class RunnerTests(unittest.TestCase):
    def exercise(self, fail_profile):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("matmul.cu", "baseline.cuh", "Makefile", "scripts/baseline.py",
                         "docs/T4.md", "matmul"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n")
            output = root / "run"

            def fake_capture(command, path, required=True):
                if command[0] == str(root / "matmul"):
                    n, iterations, samples = int(command[1]), int(command[3]), int(command[4])
                    text = ",".join(baseline.FIELDS) + "\n"
                    for kernel in ("naive", "tiled", "regtiled", "cublas"):
                        for sample in range(samples):
                            text += (f"{kernel},{n},{sample},{iterations},1,"
                                     f"{2*n**3/1e9},true\n")
                else:
                    text = "tool log\n"
                path.write_text(text)
                if command[0] == "ncu" and "--set" in command and fail_profile:
                    raise RuntimeError("profiling permission denied")
                return text

            argv = ["baseline.py", "--sizes", "32", "64", "--samples", "2", "--ncu",
                    "--output", str(output)]
            with (patch.object(baseline, "ROOT", root),
                  patch.object(baseline.sys, "argv", argv),
                  patch.object(baseline, "capture", side_effect=fake_capture),
                  patch.object(baseline.subprocess, "Popen", side_effect=OSError("no telemetry"))):
                if fail_profile:
                    with self.assertRaisesRegex(RuntimeError, "permission denied"):
                        baseline.main()
                else:
                    baseline.main()
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "failed" if fail_profile else "complete")
            self.assertTrue((output / "summary.csv").is_file())
            self.assertTrue((output / "source" / "matmul.cu").is_file())
            self.assertEqual(len(metadata["sha256"]), 6)
            self.assertEqual(len((output / "samples.csv").read_text().splitlines()), 17)

    def test_runner_collects_artifacts(self):
        self.exercise(fail_profile=False)

    def test_profile_failure_preserves_timing_and_metadata(self):
        self.exercise(fail_profile=True)


if __name__ == "__main__":
    unittest.main()
