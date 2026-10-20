from __future__ import annotations

import argparse
import json
import sys

from .data_validation import validate_csv
from .intensity import IntensityEstimator
from .normalization import HinglishNormalizer
from .ollama_reviewer import DEFAULT_OLLAMA_MODEL, OllamaReviewer
from .pipeline import EmotionPipeline
from .transformer_classifier import LocalTransformerClassifier
from .emotions import EmotionDetector, LocalEmotionClassifier
from .emotion_data import read_emotion_csv, evaluate_emotions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local Hinglish emotion pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize = subparsers.add_parser("normalize", help="Normalize a Hinglish message")
    normalize.add_argument("text")

    intensity = subparsers.add_parser("intensity", help="Estimate transparent valence/arousal cues")
    intensity.add_argument("text")

    analyze = subparsers.add_parser("analyze", help="Run the local pipeline")
    analyze.add_argument("text")
    analyze.add_argument("--reviewer", choices=("none", "ollama"), default="none")
    analyze.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL)
    analyze.add_argument("--model-path", help="Local fine-tuned transformer directory")

    validate = subparsers.add_parser("validate-data", help="Validate a training CSV")
    validate.add_argument("path")
    emotion = subparsers.add_parser("emotions", help="Detect seven emotions in Hindi-English or French-English")
    emotion.add_argument("text")
    emotion.add_argument("--language-pair", choices=("hi-en", "fr-en"), default="hi-en")
    emotion.add_argument("--model-path", help="Local seven-emotion checkpoint; otherwise use demo rules")
    emotion_eval = subparsers.add_parser("evaluate-emotions", help="Evaluate emotion accuracy overall and by language mix")
    emotion_eval.add_argument("path")
    emotion_eval.add_argument("--model-path")
    batch = subparsers.add_parser("predict-file", help="Predict CSV/XLSX and score any supplied seven-emotion labels")
    batch.add_argument("path")
    batch.add_argument("--language-pair", choices=("hi-en","fr-en"), default="hi-en")
    batch.add_argument("--model-path")
    batch.add_argument("--demo", action="store_true")
    batch.add_argument("--allow-fallback", action="store_true")
    batch.add_argument("--all-rows", action="store_true", help="Explicitly include training/validation rows in file scoring")
    batch.add_argument("--output", help="Save row predictions as CSV")
    batch.add_argument("--report-json", help="Save the full evaluation report")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args()
    if args.command == "normalize":
        result = HinglishNormalizer().normalize(args.text).to_dict()
    elif args.command == "intensity":
        result = IntensityEstimator().estimate(args.text).to_dict()
    elif args.command == "analyze":
        reviewer = OllamaReviewer(model=args.ollama_model) if args.reviewer == "ollama" else None
        classifier = LocalTransformerClassifier(args.model_path) if args.model_path else None
        result = EmotionPipeline(classifier=classifier, reviewer=reviewer).predict(args.text).to_dict()
    elif args.command == "predict-file":
        from pathlib import Path
        from functools import lru_cache
        import pandas as pd
        from .batch import read_frame, normalize_records, predict_dataset
        from .checkpoints import xlmr_checkpoint
        root=Path(__file__).resolve().parents[1]
        frame=read_frame(Path(args.path).read_bytes(),args.path)
        rows=normalize_records(frame.to_dict("records"),args.language_pair)
        @lru_cache(maxsize=1)
        def load(path): return EmotionDetector(LocalEmotionClassifier(path)) if path else EmotionDetector()
        def provider(pair):
            if args.demo:return load("")
            path=Path(args.model_path) if args.model_path else xlmr_checkpoint(root/"models",pair)
            if path:
                try:
                    detector=load(str(path))
                    trained=detector.classifier.metadata.get("trained_language_pairs",[])
                    if trained and pair not in trained:raise ValueError(f"Checkpoint was not trained for {pair}")
                    return detector
                except (ValueError,RuntimeError,OSError):
                    if not args.allow_fallback:raise
            if not args.allow_fallback:raise ValueError(f"No trained checkpoint for {pair}; use --demo or explicitly --allow-fallback")
            return load("")
        report=predict_dataset(rows,provider,test_only=not args.all_rows)
        if args.output:
            path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(report["predictions"]).to_csv(path,index=False)
        if args.report_json:
            path=Path(args.report_json);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
        result={key:value for key,value in report.items() if key!="predictions"}
    elif args.command in {"emotions", "evaluate-emotions"}:
        classifier = LocalEmotionClassifier(args.model_path) if args.model_path else None
        detector = EmotionDetector(classifier)
        if args.command == "emotions":
            result = detector.predict(args.text, args.language_pair).to_dict()
        else:
            result = evaluate_emotions(read_emotion_csv(args.path), detector)
        result["classifier"] = "trained checkpoint" if args.model_path else "demo lexical rules"
    else:
        result = validate_csv(args.path).to_dict()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
