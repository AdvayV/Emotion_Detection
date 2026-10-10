from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from hinglish_emotion.model_catalog import ENCODERS, checkpoint_ready


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EncoderComparisonTests(unittest.TestCase):
    def test_dry_run_lists_all_encoders_without_creating_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "labelled.csv"
            with data.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("text", "label"))
                for label in ("positive", "neutral", "negative"):
                    for index in range(6):
                        writer.writerow((f"{label} example {index}", label))
            output = root / "models"
            result = subprocess.run(
                [sys.executable, "-m", "training.compare_encoders", "--data", str(data),
                 "--output-root", str(output), "--freeze-encoder", "--dry-run"],
                cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
            )
            for encoder in ENCODERS:
                self.assertIn(encoder.model_id, result.stdout)
                self.assertIn(encoder.folder, result.stdout)
            self.assertEqual(result.stdout.count("--freeze-encoder"), 3)
            self.assertFalse(output.exists())

    def test_incomplete_training_folder_is_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "split_manifest.csv").write_text("split,text,label\n")
            self.assertFalse(checkpoint_ready(path))


if __name__ == "__main__":
    unittest.main()
