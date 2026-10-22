"""Install an exported multilingual emotion inference ZIP without replacing existing models."""
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import uuid
import zipfile
from hinglish_emotion.checkpoint_metadata import emotion_checkpoint_metadata
from hinglish_emotion.model_catalog import checkpoint_ready, ENCODER_BY_KEY

ROOT=Path(__file__).resolve().parents[1]
ALLOWED={"config.json","emotion_metadata.json","evaluation_guard.json","metrics.json","MODEL_CARD.md","tokenizer.json","tokenizer_config.json","special_tokens_map.json","added_tokens.json","sentencepiece.bpe.model","spiece.model","vocab.txt"}


def install(archive, models_root=None, target=None):
    models_root=Path(models_root or ROOT/"models").resolve()
    with zipfile.ZipFile(archive) as z:
        members=z.infolist()
        if len(members)>100 or sum(item.file_size for item in members)>5*1024**3:raise ValueError("Unexpected inference archive size")
        entries={}
        parents=set()
        for item in members:
            name=PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in item.filename or (item.external_attr>>16)&0o170000==0o120000:raise ValueError("Unsafe ZIP member")
            if item.is_dir():continue
            if len(name.parts)!=2:raise ValueError("Expected one checkpoint folder inside the export ZIP")
            parents.add(name.parts[0])
            filename=name.name
            if filename not in ALLOWED and filename!="model.safetensors" and not (filename.startswith("model-") and filename.endswith(".safetensors")) and filename!="model.safetensors.index.json":raise ValueError(f"Unexpected inference file: {filename}")
            if filename in entries:raise ValueError("Duplicate ZIP member")
            entries[filename]=item
        if len(parents)!=1 or not {"config.json","tokenizer.json"}<=set(entries):raise ValueError("Incomplete checkpoint export")
        config=json.loads(z.read(entries["config.json"]))
        metadata=config.get("emotion_metadata",{})
        base_model=metadata.get("base_model")
        encoder=next((spec for spec in ENCODER_BY_KEY.values() if spec.model_id==base_model),None)
        if encoder is None and config.get("model_type")=="xlm-roberta":
            encoder=ENCODER_BY_KEY["xlmr"]
        if encoder is None or metadata.get("training_status")!="complete":raise ValueError("Export must be a completed emotion run with recognized encoder provenance")
        expected_model_type="xlm-roberta" if encoder.key=="xlmr" else "bert"
        if config.get("model_type")!=expected_model_type:raise ValueError("Encoder provenance does not match config model_type")
        pairs=set(metadata.get("trained_language_pairs",[]))
        supported_pairs={"hi-en","fr-en"} if encoder.key!="muril" else {"hi-en"}
        if not pairs or not pairs<=supported_pairs:raise ValueError("Missing or unsupported trained language scope")
        advanced = metadata.get("architecture") == "text_emoji_multitask_v1"
        if advanced and encoder.key != "xlmr": raise ValueError("Learned emoji fusion currently requires XLM-R")
        default = "xlmr_fusion_emotion" if advanced else (f"{encoder.key}_multilingual_emotion" if len(pairs)==2 else f"{encoder.key}_{next(iter(pairs)).replace('-', '_')}_emotion")
        name=target or default
        if name not in {f"{encoder.key}_multilingual_emotion",f"{encoder.key}_hi_en_emotion",f"{encoder.key}_fr_en_emotion",f"{encoder.key}_emotion", *( ["xlmr_fusion_emotion"] if advanced else [] )}:raise ValueError("Use a supported target folder name for the detected encoder")
        destination=models_root/name
        if destination.exists():raise ValueError(f"Preserving existing model at {destination}. Choose a different models root or archive it explicitly first.")
        stage=models_root/(".import_"+uuid.uuid4().hex)
        if not stage.resolve().is_relative_to(models_root) or not destination.resolve().is_relative_to(models_root):raise ValueError("Invalid model destination")
        stage.mkdir(parents=True)
        try:
            for filename,item in entries.items():
                with z.open(item) as src,(stage/filename).open("xb") as out:shutil.copyfileobj(src,out,4*1024*1024)
            actual=emotion_checkpoint_metadata(stage)
            if actual["legacy"] or not checkpoint_ready(stage):raise ValueError("Missing usable model weights or tokenizer")
            if advanced:
                from hinglish_emotion.fusion import validate_fusion_checkpoint
                validate_fusion_checkpoint(stage)
            from safetensors import safe_open
            weight_files=list(stage.glob("*.safetensors"))
            if not weight_files:raise ValueError("SafeTensor weights are required")
            head_found=False
            head_name="classifier.out_proj.weight" if encoder.key=="xlmr" else "classifier.weight"
            for file in weight_files:
                with safe_open(file,framework="np") as weights:
                    if head_name in weights.keys():
                        head_found=True
                        if weights.get_slice(head_name).get_shape()!=[7,config.get("hidden_size")]:raise ValueError("Incorrect classification head")
            if not head_found:raise ValueError("Missing trained seven-emotion classification head")
            stage.rename(destination)
        except Exception:
            # Only remove files created in this validated staging folder.
            for file in stage.iterdir():
                if file.is_file() and file.resolve().parent==stage.resolve():file.unlink()
            stage.rmdir()
            raise
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("--models-root")
    parser.add_argument("--target")
    args=parser.parse_args()
    print(install(args.archive,args.models_root,args.target))


if __name__=="__main__":main()
