import json

import mlflow
import pytest
import torch

from src.training.baselines import run_baseline

TEXTS = {"positive": "enak banget", "negative": "parah banget", "neutral": "biasa"}


@pytest.fixture
def config(tmp_path):
    for period in ("p1", "p2", "p3"):
        for split in ("train", "val", "test"):
            path = tmp_path / "periods" / period / f"{split}.tsv"
            path.parent.mkdir(parents=True, exist_ok=True)
            lines = ["text\tlabel"] + [f"{text}\t{label}" for label, text in TEXTS.items()] * 2
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
        "mlflow_experiment": "test",
    }


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    # Without this, run files would be written to ./mlruns in the repository.
    mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())


@pytest.mark.parametrize("method", ["naive", "full"])
def test_run_baseline_returns_a_square_matrix_and_logs_it(config, tokenizer, make_model, method):
    matrix = run_baseline(config, method, make_model, tokenizer, torch.device("cpu"))

    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)
    run = mlflow.search_runs(experiment_names=["test"]).iloc[0]
    assert run["metrics.a_t3_p1"] == pytest.approx(matrix[2][0])
    assert run["metrics.a_t1_p3"] == pytest.approx(matrix[0][2])
    assert "metrics.avg_f1" in run and "metrics.forgetting" in run
    assert run["params.method"] == method


def test_run_baseline_writes_a_local_copy_of_the_result(config, tokenizer, make_model, tmp_path):
    run_baseline(config, "naive", make_model, tokenizer, torch.device("cpu"))

    result = json.loads((tmp_path / "out" / "naive_seed1.json").read_text())
    assert result["method"] == "naive"
    assert len(result["matrix"]) == 3
