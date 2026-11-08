import pytest
import torch

from src.training.train import make_loader, predict, score, train_step

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
