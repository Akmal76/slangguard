"""Prepare the abusive language data: one cleaned tweet and a label per line, split in three.

The source is the multi-label Indonesian Twitter set of Ibrohim and Budi (2019), CC BY-NC-SA 4.0.
Only its `Abusive` column is used. The text may not be redistributed, so it stays in DVC.
"""

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml

from src.data.clean import normalize_text
from src.data.split import split_train_val_test
from src.data.tsv import Row, write_tsv

LABELS = {"0": "not_abusive", "1": "abusive"}

# What the source left in the text that a real user would never type. Measured on the data:
# USER is in 58% of the tweets and 47% of them are abusive (38% overall), and the 353 tweets
# with a non-ASCII character are all not abusive. A model would learn these instead of words.
ARTIFACTS = re.compile(r"\b(?:USER|URL|RT)\b|\\n|\\x[0-9a-fA-F]{2}|[^\x00-\x7f]")


def clean_tweet(text: str) -> str:
    # Artifacts go first, because USER and URL are told apart from words only by upper case.
    return normalize_text(ARTIFACTS.sub(" ", text))


def read_abusive(path: Path) -> list[Row]:
    # The file is not UTF-8. latin-1 reads any byte, and non-ASCII characters are dropped anyway.
    with path.open(encoding="latin-1", newline="") as f:
        return [(clean_tweet(row["Tweet"]), LABELS[row["Abusive"]]) for row in csv.DictReader(f)]


def deduplicate(rows: list[Row]) -> list[Row]:
    """Keep one copy of each text, and drop empty texts and texts with different labels."""
    labels_by_text = defaultdict(set)
    for text, label in rows:
        if text:
            labels_by_text[text].add(label)
    return [
        (text, next(iter(labels))) for text, labels in labels_by_text.items() if len(labels) == 1
    ]


def prepare(config: dict) -> dict[str, int]:
    rows = deduplicate(read_abusive(Path(config["raw_dir"]) / "abusive" / "re_dataset.csv"))
    # Sorted, because the order of a dict of sets could change between runs and break the seed.
    rows.sort()
    splits = split_train_val_test(rows, config["split_ratios"], random.Random(config["seed"]))
    for name, split_rows in splits.items():
        write_tsv(split_rows, Path(config["abusive_dir"]) / f"{name}.tsv")
    return {name: len(split_rows) for name, split_rows in splits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    for name, count in prepare(config).items():
        print(f"{name}: {count} rows")


if __name__ == "__main__":
    main()
