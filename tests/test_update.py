import hashlib
import json

import pytest
import torch

import src.training.update as update_module
from src.training.train import add_lora
from src.training.update import evaluate, save_update, update

TEXTS = {"positive": "enak banget", "negative": "parah banget", "neutral": "biasa"}
LORA = {
    "r": 4,
    "alpha": 8,
    "dropout": 0.0,
    "target_modules": ["query", "value"],
    "learning_rate": 1e-2,
}


@pytest.fixture
def config(tmp_path):
    for period in ("p1", "p2", "p3"):
        for split in ("train", "val", "test"):
            path = tmp_path / "periods" / period / f"{split}.tsv"
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = ["text\tlabel"] + [
                f"{period} {text}\t{label}" for label, text in TEXTS.items()
            ] * 8
            path.write_text("\n".join(lines) + "\n")
    return {
        "periods_dir": str(tmp_path / "periods"),
        "periods": ["p1", "p2", "p3"],
        "max_length": 12,
        "batch_size": 8,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "warmup_ratio": 0.1,
        "epochs": 1,
        "max_grad_norm": 1.0,
        "seed": 1,
        "max_rows_per_split": None,
        "replay_fraction": 0.25,
        "lora": LORA,
    }


@pytest.fixture
def champion_dir(config, make_model, tmp_path):
    """A saved adapter, like the one the pipeline keeps as champion."""
    adapter = add_lora(make_model(), config)
    adapter.save_pretrained(tmp_path / "champion")
    return tmp_path / "champion"


def fingerprint(path):
    return hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest()


def test_first_update_has_no_champion_scores(config, make_model, tokenizer):
    _, metrics = update(config, 0, lambda c: make_model(), tokenizer, torch.device("cpu"))

    assert metrics["champion"] is None
    assert metrics["period"] == "p1"
    assert len(metrics["candidate"]["val"]) == 3 and len(metrics["candidate"]["test"]) == 3


def test_update_scores_champion_and_candidate_on_every_period(
    config, make_model, tokenizer, champion_dir
):
    _, metrics = update(
        config, 1, lambda c: make_model(), tokenizer, torch.device("cpu"), champion_dir
    )

    assert set(metrics["champion"]) == {"val", "test"}
    assert len(metrics["champion"]["val"]) == 3
    assert metrics["periods"] == ["p1", "p2", "p3"]
    assert metrics["train_seconds"] > 0


def test_the_champion_files_are_not_changed(config, make_model, tokenizer, champion_dir):
    before = fingerprint(champion_dir)

    update(config, 1, lambda c: make_model(), tokenizer, torch.device("cpu"), champion_dir)

    assert fingerprint(champion_dir) == before


def test_the_candidate_is_a_trained_copy_of_the_champion(
    config, make_model, tokenizer, champion_dir
):
    model, _ = update(
        config, 1, lambda c: make_model(), tokenizer, torch.device("cpu"), champion_dir
    )

    weights = {n: p for n, p in model.named_parameters() if "lora_B" in n}
    assert weights and all(p.requires_grad for p in weights.values())
    # LoRA B starts at zero in the saved champion. After training it must have moved.
    assert any(p.abs().sum() > 0 for p in weights.values())


def test_the_update_trains_on_the_new_period_plus_old_train_rows_only(
    config, make_model, tokenizer, champion_dir, monkeypatch
):
    seen = {}

    def fake_train_step(model, tokenizer, train, val, config, device):
        seen["train"], seen["val"] = train, val
        return 0.5

    monkeypatch.setattr(update_module, "train_step", fake_train_step)

    update(config, 2, lambda c: make_model(), tokenizer, torch.device("cpu"), champion_dir)

    train_periods = {text.split()[0] for text, _ in seen["train"]}
    assert train_periods == {"p1", "p2", "p3"}
    assert sum(text.startswith("p3") for text, _ in seen["train"]) == 24  # all of the new period
    assert sum(not text.startswith("p3") for text, _ in seen["train"]) == 12  # 25% of 2 x 24
    assert {text.split()[0] for text, _ in seen["val"]} == {"p3"}  # validation: new period only


def test_save_update_writes_the_adapter_and_the_metrics(config, make_model, tokenizer, tmp_path):
    model, metrics = update(config, 0, lambda c: make_model(), tokenizer, torch.device("cpu"))

    save_update(model, metrics, tmp_path / "out")

    assert (tmp_path / "out" / "adapter" / "adapter_config.json").exists()
    assert json.loads((tmp_path / "out" / "metrics.json").read_text())["period"] == "p1"


def test_evaluate_returns_one_score_per_period(config, make_model, tokenizer):
    scores = evaluate(add_lora(make_model(), config), config, tokenizer, torch.device("cpu"))

    assert all(0 <= f1 <= 1 for key in ("val", "test") for f1 in scores[key])
