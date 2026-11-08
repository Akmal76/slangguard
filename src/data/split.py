"""Build the drift periods P1 to P4, each with its own train, val, and test split.

P1 is SmSA. IDSA is cut into three parts with no shared text: P2 keeps the real tweets,
P3 adds dictionary slang, and P4 adds more slang plus reversed words.
"""

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from src.data.tsv import Row, read_tsv, write_tsv


def load_slang_forms(path: Path) -> dict[str, list[str]]:
    """Map each formal word to its single-word slang forms."""
    forms = defaultdict(set)
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            slang, formal = row["slang"].strip().lower(), row["formal"].strip().lower()
            if slang and formal and slang != formal and " " not in slang + formal:
                forms[formal].add(slang)
    # Sorted lists, because set order changes between runs and would break the seed.
    return {formal: sorted(slangs) for formal, slangs in forms.items()}


def group_by_label(rows: list[Row], rng: random.Random) -> dict[str, list[Row]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[1]].append(row)
    for label in groups:
        rng.shuffle(groups[label])
    return dict(sorted(groups.items()))


def split_into_parts(rows: list[Row], n_parts: int, rng: random.Random) -> list[list[Row]]:
    """Cut rows into equal parts with the same label mix."""
    parts = [[] for _ in range(n_parts)]
    for group in group_by_label(rows, rng).values():
        for i, row in enumerate(group):
            parts[i % n_parts].append(row)
    return parts


def split_train_val_test(
    rows: list[Row], ratios: dict[str, float], rng: random.Random
) -> dict[str, list[Row]]:
    splits = {"train": [], "val": [], "test": []}
    for group in group_by_label(rows, rng).values():
        n_train = round(len(group) * ratios["train"])
        n_val = round(len(group) * ratios["val"])
        splits["train"] += group[:n_train]
        splits["val"] += group[n_train : n_train + n_val]
        splits["test"] += group[n_train + n_val :]
    return splits


def most_frequent_words(rows: list[Row], count: int, min_length: int = 3) -> set[str]:
    counts = Counter(word for text, _ in rows for word in text.split())
    candidates = [w for w, _ in counts.most_common() if len(w) >= min_length and w != w[::-1]]
    return set(candidates[:count])


def rewrite(
    text: str,
    slang_forms: dict[str, list[str]],
    rate: float,
    rng: random.Random,
    reversed_words: frozenset[str] = frozenset(),
) -> str:
    words = []
    for word in text.split():
        if word in reversed_words:
            words.append(word[::-1])
        elif word in slang_forms and rng.random() < rate:
            words.append(rng.choice(slang_forms[word]))
        else:
            words.append(word)
    return " ".join(words)


def build_periods(config: dict) -> dict[str, dict[str, list[Row]]]:
    seed = config["seed"]
    interim_dir = Path(config["interim_dir"])
    smsa = read_tsv(interim_dir / "smsa.tsv")
    idsa = read_tsv(interim_dir / "idsa.tsv")
    slang_forms = load_slang_forms(Path(config["lexicon_path"]))

    # One generator per task, so changing one step never changes the others.
    part_a, part_b, part_c = split_into_parts(idsa, 3, random.Random(f"{seed}-parts"))

    p3_rng = random.Random(f"{seed}-p3")
    p3 = [(rewrite(t, slang_forms, config["slang_rate"]["p3"], p3_rng), y) for t, y in part_b]

    p4_rng = random.Random(f"{seed}-p4")
    reversed_words = frozenset(most_frequent_words(part_c, config["reversed_word_count"]))
    p4 = [
        (rewrite(t, slang_forms, config["slang_rate"]["p4"], p4_rng, reversed_words), y)
        for t, y in part_c
    ]

    periods = {"p1": smsa, "p2": part_a, "p3": p3, "p4": p4}
    return {
        name: split_train_val_test(rows, config["split_ratios"], random.Random(f"{seed}-{name}"))
        for name, rows in periods.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    periods_dir = Path(config["periods_dir"])
    for name, splits in build_periods(config).items():
        for split, rows in splits.items():
            write_tsv(rows, periods_dir / name / f"{split}.tsv")
        sizes = ", ".join(f"{split} {len(rows)}" for split, rows in splits.items())
        print(f"{name}: {sizes}")


if __name__ == "__main__":
    main()
