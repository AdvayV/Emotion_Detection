"""Tiny network-free training integration; never a research accuracy test."""

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hinglish_emotion.tasks import EMOTION_LABELS

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "pandas", "sklearn", "tokenizers"))


@unittest.skipUnless(AVAILABLE, "Model dependencies are optional")
class TrainingResumeTests(unittest.TestCase):
    def test_epoch_resume_matches_continuous_training_and_rejects_changes(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast
        from training.train_transformer import parse_args, train
        from hinglish_emotion.emotions import EmotionDetector, LocalEmotionClassifier

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base = root / "tiny_base"
                base.mkdir()
                words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *EMOTION_LABELS, "phrase", "sample", *map(str, range(20))]
                vocab = {word: index for index, word in enumerate(words)}
                backend = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
                backend.pre_tokenizer = Whitespace()
                backend.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])])
                tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]", model_max_length=64)
                tokenizer.save_pretrained(base)
                torch.manual_seed(7)
                BertForSequenceClassification(BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=24, max_position_embeddings=64, num_labels=7)).save_pretrained(base)
                data = root / "fixture.csv"
                with data.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["row_id", "text", "emotion", "language_pair", "source"])
                    for label in EMOTION_LABELS:
                        for index in range(20):
                            writer.writerow([f"{label}_{index}", f"{label} phrase sample {index}", label, "hi-en", "synthetic_smoke_fixture"])
                def arguments(output, epochs, resume=False, batch=2):
                    argv = ["train", "--task", "emotion", "--data", str(data), "--model", str(base),
                            "--output", str(output), "--epochs", str(epochs), "--batch-size", str(batch),
                            "--gradient-accumulation", "3", "--max-length", "16", "--precision", "fp32",
                            "--device", "cpu", "--allow-cpu", "--local-files-only", "--allow-synthetic"]
                    if resume:
                        argv.append("--resume")
                    with patch("sys.argv", argv):
                        return parse_args()
                uninterrupted, interrupted = root / "uninterrupted", root / "interrupted"
                with contextlib.redirect_stdout(io.StringIO()):
                    full = train(arguments(uninterrupted, 2))
                    train(arguments(interrupted, 1))
                    resumed = train(arguments(interrupted, 2, True))
                self.assertEqual(full["history"], resumed["history"])
                left = torch.load(uninterrupted / "latest_training_state.pt", weights_only=True, map_location="cpu")
                right = torch.load(interrupted / "latest_training_state.pt", weights_only=True, map_location="cpu")
                for name, tensor in left["model"].items():
                    self.assertTrue(torch.equal(tensor, right["model"][name]), name)
                classifier = LocalEmotionClassifier(interrupted, device="cpu")
                self.assertEqual(classifier.metadata["text_mode"], "raw")
                self.assertEqual(classifier.metadata["evaluation_scope"], "synthetic_only")
                self.assertEqual(classifier.max_length, 16)
                result = EmotionDetector(classifier).predict("Je suis TRÈS triste 👩🏽‍💻 acchaaa", "fr-en")
                self.assertEqual(set(result.probabilities), set(EMOTION_LABELS))
                self.assertTrue(result.experimental)
                self.assertTrue(result.needs_context)
                self.assertTrue(EmotionDetector(classifier).predict("phrase " * 100).truncated)
                self.assertTrue((interrupted / "MODEL_CARD.md").is_file())
                self.assertEqual(json.loads((interrupted / "metrics.json").read_text())["labels"], list(EMOTION_LABELS))
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    train(arguments(interrupted, 3, True, batch=3))
                data.write_text(data.read_text().replace("phrase sample 0", "changed phrase sample 0"), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "changed"):
                    train(arguments(interrupted, 3, True))
        finally:
            torch.set_num_threads(previous_threads)
