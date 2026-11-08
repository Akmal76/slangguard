import json

import mlflow
import pytest
import torch

from src.training.run import run_method, run_name
from src.training.train import add_lora

TEXTS = {"positive": "enak banget", "negative": "parah banget", "neutral": "biasa"}
LORA = {
    "r": 4,
    "alpha": 8,
    "dropout": 0.0,
    "target_modules": ["query", "value"],
    "learning_rate": 1e-3,
}


@pytest.fixture
def config(tmp_path):
    for period in ("p1", "p2", "p3"):
        for split in ("train", "val", "test"):
            path = tmp_path / "periods" / period / f"{split}.tsv"
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = ["text\tlabel"] + [f"{text}\t{label}" for label, text in TEXTS.items()] * 4
            path.write_text("\n".join(lines) + "\n")
    return {
        "periods_dir": str(tmp_path / "periods"),
        "periods": ["p1", "p2", "p3"],
        "output_dir": str(tmp_path / "out"),
        "model_name": "tiny",
        "max_length": 8,
        "batch_size": 4,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "warmup_ratio": 0.1,
        "epochs": 1,
        "max_grad_norm": 1.0,
        "seed": 1,
        "max_rows_per_split": None,
        "replay_fraction": 0.25,
        "lora": LORA,
        "mlflow_experiment": "test",
    }


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    # Without this, run files would be written to ./mlruns in the repository.
    mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())


@pytest.fixture
def make_for(make_model):
    """Returns the right model factory for a method, like main() does."""

    def pick(method):
        if method.startswith("lora"):
            return lambda config: add_lora(make_model(), config)
        return lambda config: make_model()

    return pick


def test_run_name_shows_the_replay_size_only_for_replay_methods(config):
    assert run_name(config, "naive") == "naive-seed1"
    assert run_name(config, "replay") == "replay-r25-seed1"
    assert run_name(config, "lora") == "lora-seed1"
    assert run_name(config, "lora_replay") == "lora_replay-r25-seed1"


@pytest.mark.parametrize("method", ["naive", "full", "replay", "lora", "lora_replay"])
def test_run_method_returns_a_square_matrix_and_logs_it(config, tokenizer, make_for, method):
    matrix = run_method(config, method, make_for(method), tokenizer, torch.device("cpu"))

    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)
    run = mlflow.search_runs(experiment_names=["test"]).iloc[0]
    assert run["metrics.a_t3_p1"] == pytest.approx(matrix[2][0])
    assert run["metrics.a_t1_p3"] == pytest.approx(matrix[0][2])
    assert "metrics.avg_f1" in run and "metrics.forgetting" in run
    assert run["params.method"] == method
    assert run["tags.mlflow.runName"] == run_name(config, method)


def test_lora_run_logs_lora_settings_and_a_small_trainable_share(
    config, tokenizer, make_for, tmp_path
):
    run_method(config, "lora_replay", make_for("lora_replay"), tokenizer, torch.device("cpu"))

    run = mlflow.search_runs(experiment_names=["test"]).iloc[0]
    assert run["params.lora_r"] == "4"
    assert run["params.replay_fraction"] == "0.25"
    assert float(run["params.learning_rate"]) == 1e-3
    assert int(run["params.trainable_params"]) < int(run["params.total_params"]) / 2
    assert (
        tmp_path / "out" / "adapters" / "lora_replay-r25-seed1" / "adapter_config.json"
    ).exists()


def test_full_fine_tuning_run_logs_replay_zero_and_saves_no_adapter(
    config, tokenizer, make_for, tmp_path
):
    run_method(config, "naive", make_for("naive"), tokenizer, torch.device("cpu"))

    run = mlflow.search_runs(experiment_names=["test"]).iloc[0]
    assert run["params.replay_fraction"] == "0"
    assert run["params.trainable_params"] == run["params.total_params"]
    assert not (tmp_path / "out" / "adapters").exists()


def test_run_method_writes_a_local_copy_of_the_result(config, tokenizer, make_for, tmp_path):
    run_method(config, "replay", make_for("replay"), tokenizer, torch.device("cpu"))

    result = json.loads((tmp_path / "out" / "replay-r25-seed1.json").read_text())
    assert result["run"] == "replay-r25-seed1"
    assert result["replay_fraction"] == 0.25
    assert len(result["matrix"]) == 3
    assert len(result["step_seconds"]) == 3
    assert result["total_train_seconds"] == pytest.approx(sum(result["step_seconds"]))
