from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import XLMRobertaConfig
from hinglish_emotion.fusion import (ARCHITECTURE, EMOJI_BUCKETS, MAX_EMOJI, TextEmojiFusionModel,
                                     emoji_sequence_ids, multitask_loss, LocalFusionClassifier)
from hinglish_emotion.tasks import EMOTION_LABELS


def tiny_model():
    config = XLMRobertaConfig(vocab_size=40, hidden_size=16, num_hidden_layers=1,
                             num_attention_heads=2, intermediate_size=32, max_position_embeddings=64,
                             hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
                             id2label=dict(enumerate(EMOTION_LABELS)),
                             label2id={label: index for index, label in enumerate(EMOTION_LABELS)})
    config.fusion_architecture = ARCHITECTURE
    config.emoji_hash_version = "sha256_sequence_v1"
    config.emoji_buckets = EMOJI_BUCKETS
    config.max_emoji_sequences = MAX_EMOJI
    return TextEmojiFusionModel(config)


def test_learned_gate_and_encoder_receive_gradients():
    torch.manual_seed(3)
    model = tiny_model()
    inputs = torch.tensor([[0, 4, 8, 2], [0, 5, 9, 2]])
    output = model(inputs, attention_mask=torch.ones_like(inputs), emoji_ids=torch.tensor([[12, 0], [24, 25]]))
    loss = multitask_loss(output, torch.tensor([0, 1]), torch.tensor([1, -100]))
    loss.backward()
    for parameter in (model.gate.weight, model.emoji_embeddings.weight,
                      model.encoder.embeddings.word_embeddings.weight, model.sarcasm_head.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert output.logits.shape == (2, 7)


def test_missing_emoji_is_exact_zero_contribution():
    model = tiny_model().eval()
    inputs = torch.tensor([[0, 4, 8, 2], [0, 5, 9, 2]])
    with torch.no_grad():
        # A trained projection/gate cannot create an emoji signal from padding.
        model.gate.bias.fill_(10)
        output = model(inputs, emoji_ids=torch.zeros((2, 3), dtype=torch.long))
    assert torch.count_nonzero(output.emoji_gate) == 0
    assert torch.equal(output.fused_state, output.text_state)


def test_unlabelled_sarcasm_rows_have_zero_auxiliary_gradient():
    auxiliary = torch.tensor([0.4, -0.4], requires_grad=True)
    output = SimpleNamespace(logits=torch.randn(2, 7, requires_grad=True), sarcasm_logits=auxiliary)
    multitask_loss(output, torch.tensor([0, 1]), torch.tensor([-100, 1])).backward()
    assert auxiliary.grad[0] == 0
    assert auxiliary.grad[1] != 0
    model = tiny_model()
    output = model(torch.tensor([[0, 4, 2]]), emoji_ids=torch.tensor([[12]]))
    multitask_loss(output, torch.tensor([0]), torch.tensor([-100])).backward()
    assert torch.count_nonzero(model.sarcasm_head.weight.grad) == 0


def test_safetensors_save_load_roundtrip_without_download(tmp_path, monkeypatch):
    from transformers import AutoModel
    def no_download(*args, **kwargs):
        raise AssertionError("Local roundtrip must not download a pretrained model")
    monkeypatch.setattr(AutoModel, "from_pretrained", no_download)
    model = tiny_model().eval()
    inputs = torch.tensor([[0, 4, 8, 2]])
    emoji_ids = torch.tensor([[12, 24]])
    before = model(inputs, emoji_ids=emoji_ids).logits
    model.save_pretrained(tmp_path)
    loaded = TextEmojiFusionModel.from_pretrained(tmp_path).eval()
    assert (tmp_path / "model.safetensors").is_file()
    assert not list(tmp_path.glob("*.bin"))
    assert torch.equal(before, loaded(inputs, emoji_ids=emoji_ids).logits)


def test_unicode_emoji_sequence_hashing_is_stable():
    pytest.importorskip("emoji")
    text = "hello 👩🏽‍💻 🇮🇳 ❤️"
    ids = emoji_sequence_ids(text)
    assert len(ids) == 3
    assert ids == emoji_sequence_ids(text)
    assert all(0 < value < EMOJI_BUCKETS for value in ids)
    assert emoji_sequence_ids("plain text") == []


def test_untrained_sarcasm_head_is_never_returned():
    classifier = LocalFusionClassifier.__new__(LocalFusionClassifier)
    classifier.metadata = {"trained_sarcasm_head": False}
    with pytest.raises(ValueError, match="no supervised sarcasm head"):
        classifier.predict_sarcasm("anything")

def test_context_is_ignored_with_warning_for_untrained_profile():
    class Tokenizer:
        bos_token_id = 0
        eos_token_id = 2
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[4, 8] for _ in texts]}
        def num_special_tokens_to_add(self, pair=False):
            return 4 if pair else 2
        def prepare_for_model(self, ids, pair_ids=None, **kwargs):
            ids = [0] + ids + [2] + ([2] + pair_ids + [2] if pair_ids is not None else [])
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        def pad(self, rows, **kwargs):
            return {key: torch.tensor([row[key] for row in rows]) for key in rows[0]}
    classifier = LocalFusionClassifier.__new__(LocalFusionClassifier)
    classifier.metadata = {"trained_with_context": False, "trained_sarcasm_head": False,
                           "text_mode": "raw", "preparation_version": 1, "evaluation_scope": "synthetic_only"}
    classifier.max_length = 32
    classifier.device = "cpu"
    classifier.tokenizer = Tokenizer()
    classifier.model = tiny_model().eval()
    classifier.labels = list(EMOTION_LABELS)
    before = classifier.predict_text("plain message")
    after = classifier.predict_with_context("plain message", "some context")
    assert before.probabilities == after.probabilities
    assert any("Context was ignored" in warning for warning in after.warnings)
    assert any("human-labelled" in warning for warning in after.warnings)

def test_context_budget_preserves_current_text_and_bounds_overlong_target():
    from hinglish_emotion.fusion import encode_fusion_inputs
    class Tokenizer:
        bos_token_id = 0
        eos_token_id = 2
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[int(value) for value in text.split()] for text in texts]}
        def num_special_tokens_to_add(self, pair=False):
            return 4 if pair else 2
        def prepare_for_model(self, ids, pair_ids=None, **kwargs):
            values = [0] + ids + [2] + ([2] + pair_ids + [2] if pair_ids is not None else [])
            return {"input_ids": values, "attention_mask": [1] * len(values)}
        def pad(self, rows, **kwargs):
            width = max(len(row["input_ids"]) for row in rows)
            return {key: torch.tensor([row[key] + [0] * (width-len(row[key])) for row in rows]) for key in rows[0]}
    tokenizer = Tokenizer()
    batch, lengths = encode_fusion_inputs(tokenizer, ["4 5 6", "4 5 6 7 8 9"], ["20 21 22 23", "20 21"], 8)
    assert batch["input_ids"][0].tolist() == [0, 4, 5, 6, 2, 2, 20, 2]
    assert batch["input_ids"][1].tolist() == [0, 4, 5, 6, 7, 2, 2, 2]
    assert lengths == [11, 12]
    assert batch["input_ids"].shape == (2, 8)
