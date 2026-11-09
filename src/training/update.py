"""One update step of the pipeline: teach the champion model a new period.

    python -m src.training.update --period p4 --champion outputs/c3/adapter --out outputs/c4

The champion is the adapter that is live now. The update trains a copy of it on the new
period plus a small replay sample of the older ones, then scores the champion and the new
adapter (the candidate) on the validation and test sets of every period. The pipeline
decides from the validation scores whether the candidate replaces the champion.
Without --champion, a new adapter is created. That is how the first champion is made.
"""

import argparse
import json
import time
from pathlib import Path

import yaml
from peft import PeftModel
from transformers import AutoTokenizer

from src.training.data import load_split, period_test_rows, train_rows, val_rows
from src.training.train import (
    add_lora,
    new_model,
    pick_device,
    score,
    set_seed,
    train_step,
)

METHOD = "lora_replay"


def evaluate(model, config: dict, tokenizer, device) -> dict[str, list[float]]:
    """Macro-F1 on the validation and the test set of every period."""
    periods = config["periods"]
    return {
        "val": [
            score(model, load_split(config, p, "val"), tokenizer, config, device) for p in periods
        ],
        "test": [
            score(model, period_test_rows(config, p), tokenizer, config, device) for p in periods
        ],
    }


def update(config: dict, t: int, make_base, tokenizer, device, champion_dir=None):
    """Train step t. Returns the candidate model and a dict of metrics.

    `make_base` builds a fresh pretrained model. The champion is read from `champion_dir`
    and is never written to, so it stays available if the candidate is rejected.
    """
    set_seed(config["seed"])
    base = make_base(config)
    if champion_dir is None:
        model, champion_scores = add_lora(base, config), None
    else:
        model = PeftModel.from_pretrained(base, champion_dir, is_trainable=True)
        # Loading puts the model on the CPU. Move it first: on a GPU the champion is scored
        # before train_step, which is the first place that moves the model.
        model.to(device)
        # Scored before any training, so these are the champion's own numbers.
        champion_scores = evaluate(model, config, tokenizer, device)

    train = train_rows(config, METHOD, t)
    val = val_rows(config, METHOD, t)
    start = time.perf_counter()
    best_val_f1 = train_step(model, tokenizer, train, val, config, device)
    seconds = time.perf_counter() - start

    metrics = {
        "period": config["periods"][t],
        "periods": config["periods"],
        "train_rows": len(train),
        "best_val_f1": best_val_f1,
        "train_seconds": seconds,
        "champion": champion_scores,
        "candidate": evaluate(model, config, tokenizer, device),
    }
    return model, metrics


def save_update(model, metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "adapter")  # about 1.2 MB: LoRA matrices and the head
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--period", required=True, help="the new period, for example p4")
    parser.add_argument("--champion", type=Path, help="adapter folder of the current champion")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    t = config["periods"].index(args.period)
    if (args.champion is None) != (t == 0):
        raise SystemExit(
            "Give --champion for every period except the first, and not for the first."
        )

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model, metrics = update(config, t, new_model, tokenizer, pick_device(), args.champion)
    save_update(model, metrics, args.out)
    cand = metrics["candidate"]["val"]
    print(f"{args.period}: candidate val F1 per period {[round(x, 3) for x in cand[: t + 1]]}")


if __name__ == "__main__":
    main()
