import pytest
import torch

from src.training.train import (
    add_lora,
    learning_rate,
    make_loader,
    predict,
    score,
    train_step,
    trainable_params,
)

ROWS = [
    ("enak banget", "positive"),
    ("parah banget", "negative"),
    ("biasa", "neutral"),
    ("enak", "positive"),
]
CONFIG = {
    "max_length": 8,
    "batch_size": 2,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "warmup_ratio": 0.1,
    "epochs": 2,
    "max_grad_norm": 1.0,
    "lora": {
        "r": 4,
        "alpha": 8,
        "dropout": 0.0,
        "target_modules": ["query", "value"],
        "learning_rate": 1e-2,
    },
}


@pytest.fixture
def model(make_model):
    return make_model()


def test_loader_pads_a_batch_to_its_longest_text(tokenizer):
    loader = make_loader(ROWS, tokenizer, CONFIG, shuffle=False)

    encoded, labels = next(iter(loader))

    # "[CLS] enak banget [SEP]" is 4 tokens and "[CLS] parah banget [SEP]" is 4 as well.
    assert encoded["input_ids"].shape == (2, 4)
    assert labels.tolist() == [2, 0]  # positive, negative


def test_predict_returns_one_label_per_row(tokenizer, model):
    loader = make_loader(ROWS, tokenizer, CONFIG, shuffle=False)

    predictions = predict(model, loader, torch.device("cpu"))

    assert len(predictions) == len(ROWS)
    assert set(predictions) <= {"negative", "neutral", "positive"}


def test_train_step_updates_weights_and_returns_the_best_validation_f1(tokenizer, model):
    before = model.classifier.weight.detach().clone()

    best_f1 = train_step(model, tokenizer, ROWS, ROWS, CONFIG, torch.device("cpu"))

    assert not torch.equal(before, model.classifier.weight)
    assert best_f1 == pytest.approx(score(model, ROWS, tokenizer, CONFIG, torch.device("cpu")))


@pytest.fixture
def lora_model(make_model):
    return add_lora(make_model(), CONFIG)


def test_lora_trains_only_a_few_weights(model, lora_model):
    trainable, total = trainable_params(lora_model)

    assert trainable_params(model)[0] == trainable_params(model)[1]
    assert 0 < trainable < total / 2


def test_lora_uses_its_own_learning_rate(model, lora_model):
    assert learning_rate(lora_model, CONFIG) == 1e-2
    assert learning_rate(model, CONFIG) == CONFIG["learning_rate"]


def test_lora_training_changes_adapter_and_head_but_not_frozen_weights(tokenizer, lora_model):
    before = {n: p.detach().clone() for n, p in lora_model.named_parameters()}

    train_step(lora_model, tokenizer, ROWS, ROWS, CONFIG, torch.device("cpu"))

    changed = {
        n for n, p in lora_model.named_parameters() if not torch.equal(before[n], p.detach())
    }
    assert any("lora_B" in n for n in changed)
    assert any("classifier" in n for n in changed)
    assert all("lora_" in n or "classifier" in n for n in changed)
    assert any("query.base_layer.weight" in n for n in before)  # the frozen weights exist


def test_lora_train_step_returns_the_best_validation_f1(tokenizer, lora_model):
    best_f1 = train_step(lora_model, tokenizer, ROWS, ROWS, CONFIG, torch.device("cpu"))

    assert best_f1 == pytest.approx(score(lora_model, ROWS, tokenizer, CONFIG, torch.device("cpu")))
