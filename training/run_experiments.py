"""Run explicit input/reviewer ablations against a labelled CSV."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd

from hinglish_emotion.emotions import EmotionDetector, LocalEmotionClassifier
from hinglish_emotion.experiments import VARIANTS, run_experiments
from hinglish_emotion.checkpoints import emotion_checkpoint
from hinglish_emotion.model_catalog import ENCODERS, checkpoint_path, checkpoint_ready


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--encoder", choices=["demo", *[e.key for e in ENCODERS]], default="xlmr")
    parser.add_argument("--language", choices=["hi-en", "fr-en"], default="hi-en")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["baseline", "normalized", "emoji_removed"])
    parser.add_argument("--review-mode", choices=["none", "selective", "all"], default="none")
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("experiment_results"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    reviewer=None
    if args.review_mode != "none":
        from hinglish_emotion.ollama_emotion_reviewer import OllamaEmotionReviewer
        reviewer=OllamaEmotionReviewer()
    if not args.data.is_file(): parser.error(f"Dataset not found: {args.data}")
    frame = pd.read_csv(args.data).fillna("")
    if "text" not in frame: parser.error("CSV must contain a text column")
    if "language_pair" not in frame: frame["language_pair"] = args.language
    if "split" not in frame: parser.error("CSV must declare split; experiments default to test rows only")
    if args.encoder == "demo": detector = EmotionDetector()
    else:
        spec = next(item for item in ENCODERS if item.key == args.encoder)
        path = emotion_checkpoint(args.model_root,args.language,spec.key)
        if path is None or not checkpoint_ready(path): parser.error(f"No complete local checkpoint for {args.encoder}: {path}")
        detector = EmotionDetector(LocalEmotionClassifier(path))
    frame=frame[frame["language_pair"].eq(args.language)].copy()
    result = run_experiments(frame.to_dict("records"), detector, args.variants, row_limit=args.limit,review_mode=args.review_mode,reviewer=reviewer)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    flat = [{**row, "scores": json.dumps(row["scores"], ensure_ascii=False)} for row in result["rows"]]
    pd.DataFrame(flat).to_csv(args.output / "rows.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"rows": len(flat), "summaries": result["summaries"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(main())
