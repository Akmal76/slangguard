"""Run one method over all periods and log the result matrix to MLflow.

    python -m src.training.run --method lora_replay --replay 0.05 --seed 42

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
from src.training.data import METHODS, get_method, period_test_rows, train_rows, val_rows
from src.training.train import (
    learning_rate,
    new_lora_model,
    new_model,
    pick_device,
    score,
    set_seed,
    train_step,
    trainable_params,
)


def run_name(config: dict, method: str) -> str:
    """For example `naive-seed42` or `lora_replay-r5-seed42` (5% replay)."""
    parts = [method]
    if get_method(method).replay == "buffer":
        parts.append(f"r{round(config['replay_fraction'] * 100)}")
    return "-".join(parts + [f"seed{config['seed']}"])


def run_method(config: dict, method: str, make_model, tokenizer, device) -> Matrix:
    """Train step by step and return the matrix a[t][i] (see src/evaluation/metrics.py)."""
    periods = config["periods"]
    seed = config["seed"]
    name = run_name(config, method)
    spec = get_method(method)
    mlflow.set_experiment(config["mlflow_experiment"])

    set_seed(seed)
    model = make_model(config)  # methods without a fresh model keep this one for all steps
    matrix: Matrix = []
    step_seconds: list[float] = []

    with mlflow.start_run(run_name=name):
        trainable, total = trainable_params(model)
        params = {
            key: config[key]
            for key in (
                "model_name",
                "max_length",
                "batch_size",
                "epochs",
                "warmup_ratio",
                "seed",
                "max_rows_per_split",
            )
        }
        params |= {
            "method": method,
            "learning_rate": learning_rate(model, config),
            "replay": spec.replay,
            "replay_fraction": config["replay_fraction"] if spec.replay == "buffer" else 0,
            "trainable_params": trainable,
            "total_params": total,
        }
        if spec.lora:
            params |= {f"lora_{k}": v for k, v in config["lora"].items() if k != "learning_rate"}
        mlflow.log_params(params)

        for t, period in enumerate(periods):
            if spec.fresh_model:
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
            step_seconds.append(seconds)

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
            "total_train_seconds": sum(step_seconds),
        }
        mlflow.log_metrics(summary)
        result = {
            "run": name,
            "method": method,
            "seed": seed,
            "replay_fraction": params["replay_fraction"],
            "trainable_params": trainable,
            "matrix": matrix,
            "step_seconds": step_seconds,
            **summary,
        }
        mlflow.log_dict(result, "result.json")

    # A local copy, so a lost connection to the MLflow server loses nothing.
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(result, indent=2))
    if spec.lora:
        model.save_pretrained(out / "adapters" / name)  # adapter and head only, about 1 MB
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list(METHODS), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--seed", type=int, help="overrides the seed in the config")
    parser.add_argument("--replay", type=float, help="overrides replay_fraction in the config")
    args = parser.parse_args()

    # MLflow no longer accepts a plain folder (mlruns/) as storage, so the local default
    # is a SQLite file.
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    config = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        config["seed"] = args.seed
    if args.replay is not None:
        config["replay_fraction"] = args.replay

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    make_model = new_lora_model if get_method(args.method).lora else new_model
    run_method(config, args.method, make_model, tokenizer, pick_device())


if __name__ == "__main__":
    main()
