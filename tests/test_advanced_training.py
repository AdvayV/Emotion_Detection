import builtins
import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from hinglish_emotion.tasks import EMOTION_LABELS
from training.train_advanced import parse_args, prepare_inputs, supervision_frame, train


def dataset(path, context=False, sarcasm=False):
    rows = []
    for split in ("train", "validation", "test"):
        for label in EMOTION_LABELS:
            for index in range(2):
                row = {"text": f"{split} example {label} word{chr(97 + index)}", "emotion": label,
                       "language_pair": "hi-en", "split": split,
                       "row_id": f"{split}_{label}_{index}", "group_id": f"{split}_{label}_{index}",
                       "source": "test_fixture", "annotation_status": "human_adjudicated"}
                if context:
                    row.update(context=f"preceding {split} {label}", human_reviewed="true")
                if sarcasm:
                    row.update(sarcasm=str(index), human_reviewed="true")
                rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_dry_run_checks_splits_before_torch_cuda_or_download(tmp_path, monkeypatch):
    path = dataset(tmp_path / "rows.csv")
    args = parse_args(["--data", str(path), "--output", str(tmp_path / "run"), "--dry-run"])
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers"}:
            raise AssertionError("dry-run imported the model runtime")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = train(args)
    assert result["dry_run"] is True
    assert [counts["rows"] for counts in result["supervision_counts"].values()] == [14, 14, 14]
    assert result["supervision_counts"]["train"]["sarcasm_rows"] == 0
    assert (tmp_path / "run/frozen_split.csv").is_file()
    assert not (tmp_path / "run/model.safetensors").exists()


def test_supervision_uses_only_independent_reviewed_annotations():
    frame = pd.DataFrame([{"text": "joy 😊", "label": "joy", "sarcasm": "", "human_reviewed": "true"},
                          {"text": "anger 😡", "label": "anger", "sarcasm": "1", "human_reviewed": "true"}])
    assert supervision_frame(frame, sarcasm_column="sarcasm")["sarcasm_target"].tolist() == [-100, 1]
    frame.loc[1, "human_reviewed"] = "false"
    with pytest.raises(ValueError, match="human_reviewed"):
        supervision_frame(frame, sarcasm_column="sarcasm")
    frame.loc[1, "human_reviewed"] = "true"
    frame.loc[1, "sarcasm"] = "joy"
    with pytest.raises(ValueError, match="Sarcasm labels"):
        supervision_frame(frame, sarcasm_column="sarcasm")


def test_context_requires_reviewed_labels_and_conversation_groups():
    frame = pd.DataFrame([{"text": "great", "context": "bad news", "human_reviewed": "true"}])
    with pytest.raises(ValueError, match="group_id"):
        supervision_frame(frame, use_context=True)
    frame["group_id"] = "conversation_a"
    assert supervision_frame(frame, use_context=True)["context_text"].tolist() == ["bad news"]
    frame["human_reviewed"] = "false"
    with pytest.raises(ValueError, match="human_reviewed"):
        supervision_frame(frame, use_context=True)


def test_optional_tasks_preserve_frozen_dataset_provenance(tmp_path):
    path = dataset(tmp_path / "rows.csv", context=True, sarcasm=True)
    args = parse_args(["--data", str(path), "--output", str(tmp_path / "run"),
                       "--use-context", "--sarcasm-column", "sarcasm", "--dry-run"])
    first = prepare_inputs(args)
    second = prepare_inputs(args)
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["supervision_counts"]["train"] == {"rows": 14, "context_rows": 14, "sarcasm_rows": 14}
    data = pd.read_csv(path, keep_default_na=False)
    data.loc[0, "context"] = "changed context"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="changed"):
        prepare_inputs(args)


def test_tensor_optimizer_and_rng_roundtrip_without_pickle(tmp_path):
    torch = pytest.importorskip("torch")
    from training.train_advanced import save_tensor_tree, load_tensor_tree
    parameter = torch.nn.Parameter(torch.tensor([1., 2.]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    parameter.sum().backward()
    optimizer.step()
    state = {"optimizer": optimizer.state_dict(), "random": (1, [torch.tensor([3, 4])]), "completed_epoch": 1}
    save_tensor_tree(tmp_path, state)
    loaded = load_tensor_tree(tmp_path)
    assert loaded["completed_epoch"] == 1
    assert isinstance(loaded["random"], tuple)
    assert torch.equal(loaded["optimizer"]["state"][0]["exp_avg"], state["optimizer"]["state"][0]["exp_avg"])
    assert torch.equal(loaded["random"][1][0], torch.tensor([3, 4]))
    assert not list(tmp_path.glob("*.pt"))
    restored = torch.optim.AdamW([parameter], lr=0.01)
    restored.load_state_dict(loaded["optimizer"])
    assert torch.equal(restored.state[parameter]["exp_avg"], optimizer.state[parameter]["exp_avg"])


def test_notebook_source_is_valid_and_does_not_claim_bootstrap_auxiliary_labels():
    from training.package_advanced import notebook
    book = notebook()
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook_cell", "exec")
    source = "\n".join("".join(cell["source"]) for cell in book["cells"])
    assert "USE_CONTEXT = False" in source
    assert "SARCASM_COLUMN = None" in source
    assert '"--dry-run"' in source
    assert "16000, 2000, 2000" in source
    assert "sha256" in source

def test_training_refuses_cpu_before_pretrained_loading(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from hinglish_emotion.fusion import TextEmojiFusionModel
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    def no_download(*args, **kwargs):
        raise AssertionError("A CPU-only runtime must not download or create a pretrained model")
    monkeypatch.setattr(TextEmojiFusionModel, "from_base_pretrained", no_download)
    path = dataset(tmp_path / "rows.csv")
    args = parse_args(["--data", str(path), "--output", str(tmp_path / "run")])
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        train(args)

def test_kaggle_alternative_has_no_colab_apis_and_supports_saved_output_recovery():
    from training.package_advanced import kaggle_notebook
    book = kaggle_notebook()
    sources = []
    for cell in book["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code":
            compile(source, "kaggle_cell", "exec")
            sources.append(source)
    code = "\n".join(sources)
    assert "google.colab" not in code and "files.upload" not in code and "drive.mount" not in code
    assert "/kaggle/input" in code and "/kaggle/working" in code
    assert "RESTORE_RUN_FROM" in code and "ALLOW_SYNTHETIC" in code
    assert "training.train_advanced" in code
