"""Choose which rows each method trains, validates and tests on.

Step t (counted from 0) means the model has seen periods 0..t. Every method trains on the
new period t, and the methods differ in what else they see and how they update the model:

- naive: nothing else, on top of the model from the step before.
- full: all old data, with a fresh model at every step.
- replay: a small fixed sample of each old period (the replay buffer).
- lora, lora_replay: the same as naive and replay, but only LoRA adapters are trained.
"""

import random
from dataclasses import dataclass
from pathlib import Path

from src.data.clean import LABELS
from src.data.tsv import Row, read_tsv

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}


@dataclass(frozen=True)
class Method:
    lora: bool  # train LoRA adapters instead of all weights
    replay: str  # "none", "buffer" (a sample of old data) or "all" (all old data)
    fresh_model: bool  # start from the pretrained model again at every step


METHODS = {
    "naive": Method(lora=False, replay="none", fresh_model=False),
    "full": Method(lora=False, replay="all", fresh_model=True),
    "replay": Method(lora=False, replay="buffer", fresh_model=False),
    "lora": Method(lora=True, replay="none", fresh_model=False),
    "lora_replay": Method(lora=True, replay="buffer", fresh_model=False),
}


def get_method(name: str) -> Method:
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name!r}")
    return METHODS[name]


def load_split(config: dict, period: str, split: str) -> list[Row]:
    rows = read_tsv(Path(config["periods_dir"]) / period / f"{split}.tsv")
    limit = config["max_rows_per_split"]
    if limit is None or limit >= len(rows):
        return rows
    # The split files are sorted by label, so the first rows would be one class only.
    return random.Random(config["seed"]).sample(rows, limit)


def replay_rows(config: dict, t: int) -> list[Row]:
    """A fixed sample of the train rows of every period before t.

    The sample of period i depends only on the seed and i, so it is the same at every later
    step, as a stored buffer would be. Val and test rows are never replayed.
    """
    fraction = config["replay_fraction"]
    rows = []
    for i, period in enumerate(config["periods"][:t]):
        old = load_split(config, period, "train")
        # At least one row, so a tiny fraction still replays something.
        size = max(1, round(len(old) * fraction)) if fraction > 0 else 0
        rows += random.Random(config["seed"] + i).sample(old, size)
    return rows


def train_rows(config: dict, method: str, t: int) -> list[Row]:
    replay = get_method(method).replay
    if replay == "all":
        old = [row for p in config["periods"][:t] for row in load_split(config, p, "train")]
    elif replay == "buffer":
        old = replay_rows(config, t)
    else:
        old = []
    return old + load_split(config, config["periods"][t], "train")


def val_rows(config: dict, method: str, t: int) -> list[Row]:
    # Validation only covers periods the model is allowed to know at step t, so a future
    # period never influences which epoch is kept. Only full retrain trains on all old
    # periods, so only it validates on all of them. The replay methods validate on the
    # current period alone, like naive, so a gain cannot come from better model selection.
    if get_method(method).replay == "all":
        periods = config["periods"][: t + 1]
    else:
        periods = [config["periods"][t]]
    return [row for p in periods for row in load_split(config, p, "val")]


def period_test_rows(config: dict, period: str) -> list[Row]:
    return load_split(config, period, "test")
