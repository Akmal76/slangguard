"""Prepare the real-review test set: Tokopedia product reviews from PRDECT-ID (CC BY 4.0).

It is only a test set. The model never trains or tunes on it, so it shows how the model does on
text written by real shoppers instead of the simulated periods.
"""

import argparse
import csv
from pathlib import Path

import yaml

from src.data.clean import normalize_text
from src.data.tsv import read_tsv

LABELS = {"Positive": "positive", "Negative": "negative"}


def read_reviews(path: Path) -> list[tuple[str, str, str]]:
    """Return (cleaned text, label, star rating) for each review."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [
            (
                normalize_text(row["Customer Review"]),
                LABELS[row["Sentiment"]],
                row["Customer Rating"],
            )
            for row in csv.DictReader(f)
        ]


def training_texts(periods_dir: Path) -> set[str]:
    """Every text of the train and val splits, so none of them can end up in this test set."""
    return {
        text
        for split_file in periods_dir.glob("p*/*.tsv")
        if split_file.stem in ("train", "val")
        for text, _ in read_tsv(split_file)
    }


def prepare(config: dict) -> dict[str, int]:
    reviews = read_reviews(Path(config["raw_dir"]) / "real_reviews" / "prdect_id.csv")
    seen = training_texts(Path(config["periods_dir"]))

    # A text that appears twice keeps one copy, unless its two copies disagree: then it is dropped.
    by_text: dict[str, set[tuple[str, str]]] = {}
    for text, label, rating in reviews:
        if text:
            by_text.setdefault(text, set()).add((label, rating))
    kept = []
    for text, labels in sorted(by_text.items()):
        if len({label for label, _ in labels}) == 1 and text not in seen:
            label, rating = sorted(labels)[0]
            kept.append((text, label, rating))

    out = Path(config["real_reviews_dir"]) / "reviews.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["text\tlabel\trating", *("\t".join(row) for row in kept)]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"raw": len(reviews), "kept": len(kept)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    for name, count in prepare(config).items():
        print(f"{name}: {count} rows")


if __name__ == "__main__":
    main()
