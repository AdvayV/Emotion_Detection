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
    else:
        result = validate_csv(args.path).to_dict()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
