"""Focused UI coverage for the read-only XLM-R / MuRIL study panel."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STREAMLIT_AVAILABLE = importlib.util.find_spec("streamlit") is not None
CONTRACT_FILE = PROJECT_ROOT / "colab" / "muril_study" / "study_contract.json"
BASELINE_FILE = PROJECT_ROOT / "colab" / "muril_study" / "xlmr_baseline_metrics.json"
COMPARISON_ARCHIVE = PROJECT_ROOT / "Documents for review/Supporting results/muril_hi_en_comparison.zip"


@unittest.skipUnless(STREAMLIT_AVAILABLE, "Dashboard dependencies are optional")
class MurilStudyUITests(unittest.TestCase):
    def render(self, root: Path):
        from streamlit.testing.v1 import AppTest

        code = (
            "from pathlib import Path\n"
            "from dashboard.muril_study import render_muril_study\n"
            f"render_muril_study(Path({str(root)!r}), lambda figure, height: figure)\n"
        )
        return AppTest.from_string(code, default_timeout=30).run()

    def packaged_study(self):
        self.assertTrue(CONTRACT_FILE.is_file(), "Packaged study contract is required")
        self.assertTrue(BASELINE_FILE.is_file(), "Packaged XLM-R baseline metrics are required")
        contract = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
        baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
        return contract, baseline

    def install_study(self, root: Path, contract, baseline):
        study_dir = root / "colab" / "muril_study"
        study_dir.mkdir(parents=True, exist_ok=True)
        (study_dir / "study_contract.json").write_text(json.dumps(contract), encoding="utf-8")
        (study_dir / "xlmr_baseline_metrics.json").write_text(json.dumps(baseline), encoding="utf-8")

    def test_pending_run_shows_verified_baseline_without_a_chart(self):
        contract, baseline = self.packaged_study()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.install_study(root, contract, baseline)
            app = self.render(root)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 3)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertTrue(any("MuRIL has not been trained or installed" in item.value for item in app.info))
        self.assertTrue(any("synthetic" in item.value.lower() for item in app.warning))
        self.assertTrue(any("Verified XLM-R hi-en accuracy" == item.label for item in app.metric))

    def test_missing_or_invalid_baseline_never_displays_metrics(self):
        contract, baseline = self.packaged_study()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "colab" / "muril_study").mkdir(parents=True)
            (root / "colab" / "muril_study" / "study_contract.json").write_text(json.dumps(contract), encoding="utf-8")
            missing = self.render(root)
            self.assertEqual(len(missing.exception), 0)
            self.assertEqual(len(missing.metric), 0)
            self.assertTrue(any("unavailable" in item.value for item in missing.info))

            baseline["dataset_sha256"] = "not-the-contracted-dataset"
            (root / "colab" / "muril_study" / "xlmr_baseline_metrics.json").write_text(json.dumps(baseline), encoding="utf-8")
            invalid = self.render(root)

        self.assertEqual(len(invalid.exception), 0)
        self.assertEqual(len(invalid.metric), 0)
        self.assertTrue(any("could not be verified" in item.value for item in invalid.warning))

    def test_valid_muril_run_shows_chart_tables_and_downloads(self):
        contract, baseline = self.packaged_study()
        muril = copy.deepcopy(baseline)
        expected = contract["matched"]
        muril["base_model"] = "google/muril-base-cased"
        for key in ("dataset_sha256", "split_sha256", "split_sizes"):
            muril[key] = copy.deepcopy(expected[key])
        for key, value in expected.get("settings", {}).items():
            muril[key] = copy.deepcopy(value)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.install_study(root, contract, baseline)
            metrics_path = root / "models" / "muril_hi_en_emotion" / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(muril), encoding="utf-8")
            app = self.render(root)

        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.get("plotly_chart")), 1)
        self.assertGreaterEqual(len(app.dataframe), 2)
        self.assertTrue(any("model weights are unavailable" in item.value for item in app.info))
        download_labels = {item.label for item in app.get("download_button")}
        self.assertIn("Download comparison (CSV)", download_labels)
        self.assertIn("Download per-emotion results (CSV)", download_labels)

    def test_muril_with_a_different_split_is_refused(self):
        contract, baseline = self.packaged_study()
        muril = copy.deepcopy(baseline)
        expected = contract["matched"]
        muril["base_model"] = "google/muril-base-cased"
        for key in ("dataset_sha256", "split_sha256", "split_sizes"):
            muril[key] = copy.deepcopy(expected[key])
        for key, value in expected.get("settings", {}).items():
            muril[key] = copy.deepcopy(value)
        muril["split_sha256"] = "different-split"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.install_study(root, contract, baseline)
            metrics_path = root / "models" / "muril_hi_en_emotion" / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(muril), encoding="utf-8")
            app = self.render(root)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertTrue(any("differs from the study contract" in item.value for item in app.warning))

    def test_colab_report_displays_scores_without_checkpoint_weights(self):
        contract, baseline = self.packaged_study()
        self.assertTrue(COMPARISON_ARCHIVE.is_file(), "Packaged Colab comparison export is required")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.install_study(root, contract, baseline)
            target=root / "Documents for review/Supporting results/muril_hi_en_comparison.zip"
            target.parent.mkdir(parents=True)
            shutil.copyfile(COMPARISON_ARCHIVE, target)
            app = self.render(root)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 4)
        self.assertTrue(any(item.label == "MuRIL Hindi-English accuracy" for item in app.metric))
        self.assertTrue(any(item.label == "MuRIL Hindi-English macro-F1" for item in app.metric))
        self.assertEqual(len(app.get("plotly_chart")), 2)
        self.assertEqual(len(app.dataframe), 2)
        self.assertTrue(any("Colab results available; MuRIL checkpoint not installed" in item.value for item in app.success))
        self.assertTrue(any("muril_hi_en.zip" in item.value and "training.install_checkpoint" in item.value for item in app.info))
        self.assertFalse((root / "models" / "muril_hi_en_emotion").exists())

    def test_malformed_colab_report_shows_warning_without_charts_or_scores(self):
        contract, baseline = self.packaged_study()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.install_study(root, contract, baseline)
            (root / "muril_hi_en_comparison.zip").write_bytes(b"not a zip archive")
            app = self.render(root)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertTrue(any("Colab comparison report could not be verified" in item.value for item in app.warning))


if __name__ == "__main__":
    unittest.main()
