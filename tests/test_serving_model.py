import copy

import pytest
import torch

from src.serving.model import SentimentModel
from src.training.train import add_lora

LORA = {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["query", "value"]}
INFO = {"base_model": "tiny", "adapter_repo": "none", "adapter_revision": "none"}


@pytest.fixture
def adapter_model(make_model):
    """A tiny model with LoRA matrices and a head that are not at their start values."""
    model = add_lora(make_model(), {"lora": LORA})
    torch.manual_seed(1)
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:  # LoRA matrices and the classification head
                param.normal_(std=0.5)
    return model


def test_merged_model_gives_the_same_probabilities_as_the_adapter(adapter_model, tokenizer):
    texts = ["enak banget", "parah banget", "biasa"]
    unmerged = SentimentModel(adapter_model, tokenizer, INFO, max_length=8)
    merged = SentimentModel(copy.deepcopy(adapter_model).merge_and_unload(), tokenizer, INFO, 8)

    for a, b in zip(unmerged.predict(texts), merged.predict(texts), strict=True):
        assert a.keys() == b.keys()
        for label in a:
            assert a[label] == pytest.approx(b[label], abs=1e-5)


def test_the_merge_really_changes_the_base_weights(adapter_model, make_model):
    merged = copy.deepcopy(adapter_model).merge_and_unload()
    plain = make_model()

    assert not torch.equal(
        merged.bert.encoder.layer[0].attention.self.query.weight,
        plain.bert.encoder.layer[0].attention.self.query.weight,
    )


def test_predict_returns_one_probability_per_label_that_sums_to_one(make_model, tokenizer):
    model = SentimentModel(make_model(), tokenizer, INFO, max_length=8)

    result = model.predict(["enak banget", "biasa"])

    assert len(result) == 2
    assert list(result[0]) == ["negative", "neutral", "positive"]
    assert sum(result[0].values()) == pytest.approx(1.0)


def test_predict_normalizes_text_like_training(make_model, tokenizer):
    model = SentimentModel(make_model(), tokenizer, INFO, max_length=8)

    assert model.predict(["Enak, BANGET!"]) == model.predict(["enak banget"])


def test_predictions_do_not_depend_on_the_batch(make_model, tokenizer):
    model = SentimentModel(make_model(), tokenizer, INFO, max_length=8)

    alone = model.predict(["biasa"])[0]
    in_batch = model.predict(["enak banget parah banget enak", "biasa"])[1]

    for label in alone:
        assert alone[label] == pytest.approx(in_batch[label], abs=1e-5)
