import ast
import hashlib
import importlib.util
import inspect
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_generator():
    spec = importlib.util.spec_from_file_location("package_project", ROOT / "training/package_project.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_job_routing_and_notebook_cells():
    module = load_generator()
    assert set(module.MODELS) == {"xlmr_combined", "mbert_combined", "muril_hi_en"}
    notebook = module.notebook()
    assert notebook["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    ast.parse(code)
    assert 'JOB="xlmr_combined"' in code and '"sarcasm_hi_en"' in code
    assert "torch.cuda.is_available()" in code and "MANIFEST.json" in code
    assert "export_checkpoint" in code and "export_sarcasm_checkpoint" in code
    assert '"--resume"' in code
    assert "AUDIT=None" in code and "USE_OWN_SARCASM_DATA" not in code
    assert 'run/"models"/(model+"_emotion")' in code
    assert 'STORAGE/"sarcasm_hi_en"' in code
    assert "test_metrics" in code and 'metrics.get("test",{})' in code
    assert "resolved_package.parent != output" in inspect.getsource(module.build)
    assert "package.is_symlink()" in inspect.getsource(module.build)

def test_generated_package_hashes_and_no_weights(tmp_path):
    module = load_generator()
    result = module.build(tmp_path / "handoff")
    archive = Path(result["zip"])
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert any(n.endswith("sarcasm_template.csv") for n in names)
        assert not any(n.endswith((".safetensors", ".bin")) for n in names)
        manifest_name = next(n for n in names if n.endswith("/MANIFEST.json"))
        prefix = manifest_name.rsplit("/", 1)[0] + "/"
        manifest = json.loads(z.read(manifest_name))
        for rel, expected in manifest["sha256"].items():
            assert hashlib.sha256(z.read(prefix + rel)).hexdigest() == expected
