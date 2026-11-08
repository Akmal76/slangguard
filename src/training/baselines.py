"""Run a baseline over all periods and log the result matrix to MLflow.

    python -m src.training.baselines --method naive --config configs/train.yaml

To log to a server, set MLFLOW_TRACKING_URI (and MLFLOW_TRACKING_USERNAME and
MLFLOW_TRACKING_PASSWORD for DagsHub). Without them, runs go to a local mlflow.db file.
"""

import argparse
import json
import os
import time
from pathlib import Path

import mlflow
import yaml
from transformers import AutoTokenizer

from src.evaluation.metrics import Matrix, avg_f1, forgetting
from src.training.data import get_method, period_test_rows, train_rows, val_rows
from src.training.train import new_model, pick_device, score, set_seed, train_step


def run_baseline(config: dict, method: str, make_model, tokenizer, device) -> Matrix:
    """Train step by step and return the matrix a[t][i] (see src/evaluation/metrics.py)."""
    periods = config["periods"]
    seed = config["seed"]
    mlflow.set_experiment(config["mlflow_experiment"])

    set_seed(seed)
    model = make_model(config)  # naive keeps this one model for all steps
    matrix: Matrix = []
    total_seconds = 0.0

    with mlflow.start_run(run_name=f"{method}-seed{seed}"):
        mlflow.log_params(
            {
                key: config[key]
                for key in (
                    "model_name",
                    "max_length",
                    "batch_size",
                    "learning_rate",
                    "epochs",
                    "warmup_ratio",
                    "seed",
                    "max_rows_per_split",
                )
            }
            | {"method": method}
        )

        for t, period in enumerate(periods):
            if get_method(method).fresh_model:
                set_seed(seed)
                model = make_model(config)  # a fresh model at every step

            start = time.perf_counter()
            val_f1 = train_step(
                model,
                tokenizer,
                train_rows(config, method, t),
                val_rows(config, method, t),
                config,
                device,
            )
            seconds = time.perf_counter() - start
            total_seconds += seconds

            # Every test set is scored after every step, including periods not seen yet.
            row = [
                score(model, period_test_rows(config, p), tokenizer, config, device)
                for p in periods
            ]
            matrix.append(row)

            mlflow.log_metrics({"val_f1": val_f1, "train_seconds": seconds}, step=t + 1)
            mlflow.log_metrics({f"a_t{t + 1}_p{i + 1}": v for i, v in enumerate(row)})
            print(f"step {t + 1} ({period}): val F1 {val_f1:.3f}, test F1 {row}, {seconds:.0f}s")

        summary = {
            "avg_f1": avg_f1(matrix),
            "forgetting": forgetting(matrix),
            "total_train_seconds": total_seconds,
        }
        mlflow.log_metrics(summary)
        result = {"method": method, "seed": seed, "matrix": matrix, **summary}
        mlflow.log_dict(result, "result.json")

    # A local copy, so a lost connection to the MLflow server loses nothing.
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{method}_seed{seed}.json").write_text(json.dumps(result, indent=2))
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # LoRA methods are added to this runner in a later step.
    parser.add_argument("--method", choices=["naive", "full", "replay"], required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--seed", type=int, help="overrides the seed in the config")
    args = parser.parse_args()

    # MLflow no longer accepts a plain folder (mlruns/) as storage, so the local default
    # is a SQLite file.
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    config = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    run_baseline(config, args.method, new_model, tokenizer, pick_device())


if __name__ == "__main__":
    main()
