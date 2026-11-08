"""Choose which rows each baseline trains, validates and tests on.

Step t (counted from 0) means the model has seen periods 0..t. The two baselines differ
only in what they train on at that step:

- naive: only period t, on top of the model from the step before.
- full: periods 0..t together, with a fresh model.
"""

from pathlib import Path

from src.data.clean import LABELS
from src.data.tsv import Row, read_tsv

METHODS = ("naive", "full")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


def load_split(config: dict, period: str, split: str) -> list[Row]:
    rows = read_tsv(Path(config["periods_dir"]) / period / f"{split}.tsv")
    limit = config["max_rows_per_split"]
    return rows if limit is None else rows[:limit]


def _periods_seen(config: dict, method: str, t: int) -> list[str]:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method!r}")
    periods = config["periods"]
    return [periods[t]] if method == "naive" else periods[: t + 1]


def train_rows(config: dict, method: str, t: int) -> list[Row]:
    return [row for p in _periods_seen(config, method, t) for row in load_split(config, p, "train")]


def val_rows(config: dict, method: str, t: int) -> list[Row]:
    # Validation only covers periods the model is allowed to know at step t, so a future
    # period never influences which epoch is kept.
    return [row for p in _periods_seen(config, method, t) for row in load_split(config, p, "val")]


def period_test_rows(config: dict, period: str) -> list[Row]:
    return load_split(config, period, "test")
