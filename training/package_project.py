"""Generate the safe Colab training ZIP and configuration notebook."""
from __future__ import annotations
import argparse, hashlib, json, shutil, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"colab"/"project_training"
PACKAGE_NAME="hinglish_project_training"
MODELS={"xlmr_combined":"FacebookAI/xlm-roberta-base","mbert_combined":"google-bert/bert-base-multilingual-cased","muril_hi_en":"google/muril-base-cased"}

def notebook():
    cells=[]
    def add(kind, text):
        c={"cell_type":kind,"id":f"step_{len(cells):02d}","metadata":{},"source":text.splitlines(keepends=True)}
        if kind=="code": c.update(execution_count=None,outputs=[])
        cells.append(c)
    add("markdown", """# Hinglish emotion training on Colab\n\nRun exactly one GPU job per session. Select one JOB below. The default `xlmr_combined` trains one seven-emotion classifier on hi-en and fr-en. This is supervised task fine-tuning, not training a language model from scratch.\n""")
    add("code", '''from pathlib import Path
import hashlib, json, os, subprocess, sys, zipfile
from google.colab import files, drive
uploaded=files.upload()
zips=[n for n in uploaded if n.lower().endswith(".zip")]
assert len(zips)==1, "Upload exactly one generated training ZIP"
base=Path("/content").resolve()
with zipfile.ZipFile(zips[0]) as z:
    assert sum(i.file_size for i in z.infolist()) < 512*1024**2
    for i in z.infolist():
        assert "\\\\" not in i.filename and (base/i.filename).resolve().is_relative_to(base)
        assert (i.external_attr>>16)&0o170000 != 0o120000
    z.extractall(base)
PACKAGE=base/"hinglish_project_training"
m=json.loads((PACKAGE/"MANIFEST.json").read_text())
for name,digest in m["sha256"].items():
    p=(PACKAGE/name).resolve(); assert p.is_relative_to(PACKAGE)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest, name
PROJECT=PACKAGE/"source"; os.chdir(PROJECT)
subprocess.run([sys.executable,"-m","pip","install","-e",".[model]"],check=True)
import torch
assert torch.cuda.is_available(), "Enable Runtime > Change runtime type > T4 GPU, then restart from this cell"
print("GPU:",torch.cuda.get_device_name(0))
''')
    add("code", '''# Select exactly one job for this Colab session.
JOB="xlmr_combined"  # xlmr_combined | mbert_combined | muril_hi_en | sarcasm_hi_en
EPOCHS=3
USE_OWN_EMOTION_DATA=False
assert JOB in {"xlmr_combined","mbert_combined","muril_hi_en","sarcasm_hi_en"}
drive.mount("/content/drive")
STORAGE=Path("/content/drive/MyDrive/HinglishEmotionProject")
STORAGE.mkdir(parents=True,exist_ok=True)
DATA=PACKAGE/"data"/"all_splits.csv"
AUDIT=PACKAGE/"data"/"dataset_audit.json"
if USE_OWN_EMOTION_DATA:
    up=files.upload(); csvs=[n for n in up if n.lower().endswith(".csv")]
    assert len(csvs)==1, "Upload one reviewed canonical emotion CSV"
    DATA=Path(csvs[0]).resolve()
    AUDIT=None
''')
    add("markdown", """## Dataset and models\n\nBootstrap: synthetic, not human-reviewed; it cannot establish real-world emotion accuracy. The frozen splits are copied to Drive and reused on resume. For reviewed emotion data, supply `text,emotion,language_pair` and keep group identities within a single explicit split when supplied. Do not transform binary sentiment or emotion labels into sarcasm labels. Model cards: [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base), [mBERT](https://huggingface.co/google-bert/bert-base-multilingual-cased), [MuRIL](https://huggingface.co/google/muril-base-cased). MuRIL job is Hindi-English only.\n\nFor `sarcasm_hi_en`, upload your own independently human-labelled CSV with `text,sarcasm,language_pair,split,group_id,human_reviewed`. Labels must be `sarcastic` or `not_sarcastic` (0/1 accepted). Both classes must occur in each split; group ids and exact duplicate texts may not cross splits. Synthetic rows are refused unless explicitly opted in in the trainer CLI. The package includes a header-only template and no sarcasm examples.\n""")
    add("code", '''if JOB=="sarcasm_hi_en":
    up=files.upload(); csvs=[n for n in up if n.lower().endswith(".csv")]
    assert len(csvs)==1, "Upload your human-labelled sarcasm CSV; no fallback dataset exists"
    command=[sys.executable,"-u","-m","training.train_sarcasm","--data",str(Path(csvs[0]).resolve()),"--model","FacebookAI/xlm-roberta-base","--output",str(STORAGE/"sarcasm_hi_en"),"--epochs",str(EPOCHS),"--resume"]
else:
    configs={"xlmr_combined":("xlmr","all"),"mbert_combined":("mbert","all"),"muril_hi_en":("muril","hi-en")}
    model,language=configs[JOB]
    run=STORAGE/JOB; run.mkdir(parents=True,exist_ok=True)
    from training.data_workflow import prepare_data
    prepared=prepare_data(DATA,"emotion","raw",42,run/"frozen_split.csv",AUDIT if AUDIT and AUDIT.exists() else None,language,allow_synthetic=not USE_OWN_EMOTION_DATA)
    command=[sys.executable,"-u","-m","training.compare_encoders","--task","emotion","--models",model,"--data",str(DATA),"--language-pair",language,"--output-root",str(run/"models"),"--split-manifest",str(run/"frozen_split.csv"),"--epochs",str(EPOCHS),"--batch-size","2","--gradient-accumulation","8","--max-length","128","--learning-rate","2e-5","--precision","auto","--text-mode","raw","--seed","42","--split-seed","42","--resume"]
    if AUDIT and AUDIT.exists(): command += ["--data-audit",str(AUDIT)]
    if not USE_OWN_EMOTION_DATA: command += ["--allow-synthetic"]
subprocess.run(command,cwd=PROJECT,check=True)
''')
    add("markdown", """## Results\n\nSeven emotion checkpoints report validation selection and held-out test results. Sarcasm is a separate binary classifier and will not be merged with the emotion output. Save the full Drive folder to resume. Inference exports should include only tokenizer/config and safetensors; exclude raw rows and optimizer state.\n""")
    add("code", '''# Inspect held-out results and export the best inference checkpoint.
from training.cloud_bundle import export_checkpoint, export_sarcasm_checkpoint
if JOB=="sarcasm_hi_en":
    model_dir=STORAGE/"sarcasm_hi_en"
    metadata=json.loads((model_dir/"sarcasm_metadata.json").read_text())
    print("Task:",metadata.get("task"),"Base model:",metadata.get("base_model"))
    print("Selected validation macro-F1:",metadata.get("selected_validation_macro_f1"))
    display(metadata.get("test_metrics",{}))
    export_path=STORAGE/"exports"/"sarcasm_hi_en.zip"
    export_sarcasm_checkpoint(model_dir,export_path)
else:
    model_dir=run/"models"/(model+"_emotion")
    metadata=json.loads((model_dir/"emotion_metadata.json").read_text())
    metrics=json.loads((model_dir/"metrics.json").read_text())
    print("Task:",metadata.get("task"),"Base model:",metadata.get("base_model"))
    print("Selected validation macro-F1:",metrics.get("best_validation_macro_f1"))
    display(metrics.get("test",{}))
    export_path=STORAGE/"exports"/(JOB+".zip")
    export_checkpoint(model_dir,export_path)
print("Export:",export_path)
files.download(str(export_path))
''')
    return {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"colab":{"name":"PROJECT_TRAINING.ipynb","provenance":[]}},"nbformat":4,"nbformat_minor":5}

def build(output=None):
    output=Path(output or OUT).resolve()
    output.mkdir(parents=True,exist_ok=True)
    output=output.resolve(strict=True)
    package=output/"package"; data=ROOT/"data/training/multilingual_bootstrap_v1"
    if not data.exists(): raise FileNotFoundError(f"Run the existing multilingual bootstrap generator first: {data}")
    resolved_package=package.resolve(strict=False)
    if package.is_symlink() or resolved_package.parent != output or resolved_package == output:
        raise ValueError(f"Refusing to remove package outside output directory: {package}")
    if package.exists(): shutil.rmtree(package)
    (package/"source").mkdir(parents=True); (package/"data").mkdir()
    paths=[]
    for folder in ("hinglish_emotion","training"):
        dest=package/"source"/folder; dest.mkdir()
        for f in sorted((ROOT/folder).glob("*.py")):
            if f.name!="package_project.py": shutil.copyfile(f,dest/f.name); paths.append(dest/f.name)
    shutil.copyfile(ROOT/"pyproject.toml",package/"source/pyproject.toml"); paths.append(package/"source/pyproject.toml")
    for name in ("all_splits.csv","dataset_audit.json","DATASET_CARD.md"):
        f=data/name
        if f.exists(): shutil.copyfile(f,package/"data"/name); paths.append(package/"data"/name)
    template=package/"data/sarcasm_template.csv"; template.write_text("text,sarcasm,language_pair,split,group_id,human_reviewed,synthetic\n",encoding="utf-8-sig"); paths.append(template)
    hashes={f.relative_to(package).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    (package/"MANIFEST.json").write_text(json.dumps({"package":PACKAGE_NAME,"sha256":hashes,"contains_model_weights":False},indent=2),encoding="utf-8")
    output.mkdir(parents=True,exist_ok=True)
    (output/"PROJECT_TRAINING.ipynb").write_text(json.dumps(notebook(),indent=2,ensure_ascii=False),encoding="utf-8")
    guide="""# START HERE\n\n1. Upload `PROJECT_TRAINING.ipynb` to Google Colab and choose Runtime > Change runtime type > GPU.\n2. Run the first cell and upload `project_training.zip` when prompted. Extraction is path-checked and every packaged file is SHA-256 checked before install. No weights are in this ZIP.\n3. Keep the default `JOB=\"xlmr_combined\"` for one XLM-R seven-emotion run across Hindi-English and English-French; choose `mbert_combined` for mBERT, `muril_hi_en` for Hindi-English MuRIL, or `sarcasm_hi_en` for a separate supervised binary run. Run one job per Colab session.\n4. Approve Drive mount. Frozen splits, training states and checkpoints save under `MyDrive/HinglishEmotionProject`; rerun the same job to resume. Do not change data or settings when resuming.\n5. Sarcasm requires your own labelled dataset. Template: `package/data/sarcasm_template.csv`; rows need human review, explicit independent train/validation/test split, unique group_id and both classes in each split. No labels or corpus are provided or generated.\n6. Model cards: https://huggingface.co/FacebookAI/xlm-roberta-base ; https://huggingface.co/google-bert/bert-base-multilingual-cased ; https://huggingface.co/google/muril-base-cased\n\nEmotion bootstrap data is synthetic (see `package/data/DATASET_CARD.md`) and does not demonstrate human-world accuracy. `training/train_transformer.py` exports safetensors and emits task metadata. Inspect validation selection and test metrics before any use.\n"""
    guide += "\n\n7. After training, download the best inference export. Install an emotion export with `python -m training.install_checkpoint \"C:/path/to/xlmr_combined.zip\"`; it is placed in a distinct model folder. Install a sarcasm export with `python -m training.install_sarcasm_checkpoint \"C:/path/to/sarcasm_hi_en.zip\"`; it uses its own separate sarcasm model folder and does not replace an emotion model."
    (output/"START_HERE.md").write_text(guide,encoding="utf-8")
    archive=output/"project_training.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for f in sorted([*paths,package/"MANIFEST.json"]): z.write(f,PACKAGE_NAME+"/"+f.relative_to(package).as_posix())
    return {"zip":str(archive),"notebook":str(output/"PROJECT_TRAINING.ipynb"),"files":len(paths)}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output");a=p.parse_args();print(json.dumps(build(a.output),indent=2))
