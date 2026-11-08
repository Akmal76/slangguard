"""Clean raw SmSA and IDSA data into one labeled table per dataset."""

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

from src.data.tsv import Row, write_tsv

LABELS = ("negative", "neutral", "positive")
IDSA_LABELS = {"-1": "negative", "0": "neutral", "1": "positive"}


def normalize_text(text: str) -> str:
    # Repeated letters ("bangettt") are kept on purpose: they are part of the slang signal.
    return " ".join(text.lower().split())


def read_smsa(raw_dir: Path) -> list[Row]:
    rows = []
    for split in ("train", "valid", "test"):
        for line in (raw_dir / "smsa" / f"{split}.tsv").read_text(encoding="utf-8").splitlines():
            if line.strip():
                text, label = line.rsplit("\t", 1)
                rows.append((text, label))
    return rows


def read_idsa(raw_dir: Path) -> list[Row]:
    lines = (raw_dir / "idsa" / "tweets.tsv").read_text(encoding="utf-8").splitlines()
    rows = []
    for line in lines[1:]:
        label, text = line.split("\t", 1)
        if label not in IDSA_LABELS:
            raise ValueError(f"Unknown IDSA label: {label!r}")
        rows.append((text, IDSA_LABELS[label]))
    return rows


def deduplicate(rows: list[Row]) -> list[Row]:
    """Keep one copy of each text, and drop texts that appear with different labels."""
    labels_by_text = defaultdict(set)
    for text, label in rows:
        if label not in LABELS:
            raise ValueError(f"Unknown label: {label!r}")
        if text:
            labels_by_text[text].add(label)
    return [
        (text, next(iter(labels))) for text, labels in labels_by_text.items() if len(labels) == 1
    ]


def remove_texts(rows: list[Row], texts: set[str]) -> list[Row]:
    return [(text, label) for text, label in rows if text not in texts]


def clean_all(config: dict) -> dict[str, int]:
    raw_dir = Path(config["raw_dir"])
    interim_dir = Path(config["interim_dir"])

    smsa = deduplicate([(normalize_text(t), label) for t, label in read_smsa(raw_dir)])
    idsa = deduplicate([(normalize_text(t), label) for t, label in read_idsa(raw_dir)])
    # P1 comes from SmSA and later periods from IDSA, so they must not share any text.
    idsa = remove_texts(idsa, {text for text, _ in smsa})

    write_tsv(smsa, interim_dir / "smsa.tsv")
    write_tsv(idsa, interim_dir / "idsa.tsv")
    return {"smsa": len(smsa), "idsa": len(idsa)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    for dataset, count in clean_all(config).items():
        print(f"{dataset}: {count} rows")


if __name__ == "__main__":
    main()
