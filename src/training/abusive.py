"""Train the abusive language detector: TF-IDF over letter groups, then logistic regression.

    uv run python -m src.training.abusive

It is a second, separate model. The sentiment model says how positive a text is, and this one
says whether it contains abusive words. It runs on a laptop CPU in seconds.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.data.clean import normalize_text
from src.data.tsv import read_tsv

POSITIVE = "abusive"


def load(data_dir: Path, split: str) -> tuple[list[str], list[int]]:
    rows = read_tsv(data_dir / f"{split}.tsv")
    return [text for text, _ in rows], [int(label == POSITIVE) for _, label in rows]


def build(config: dict, c: float) -> Pipeline:
    tfidf, lr = config["tfidf"], config["logistic_regression"]
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=tfidf["analyzer"],
                    ngram_range=tuple(tfidf["ngram_range"]),
                    min_df=tfidf["min_df"],
                    sublinear_tf=tfidf["sublinear_tf"],
                ),
            ),
            (
                "model",
                LogisticRegression(
                    C=c,
                    class_weight=lr["class_weight"],
                    max_iter=lr["max_iter"],
                    random_state=config["seed"],
                ),
            ),
        ]
    )


def score(model: Pipeline, texts: list[str], labels: list[int]) -> dict[str, float]:
    predicted = model.predict(texts)
    return {
        "f1_abusive": f1_score(labels, predicted),
        "precision_abusive": precision_score(labels, predicted),
        "recall_abusive": recall_score(labels, predicted),
        "macro_f1": f1_score(labels, predicted, average="macro"),
    }


def load_lexicon(path: Path) -> list[str]:
    """The authors' list of abusive words and phrases, cleaned like the texts."""
    with path.open(encoding="latin-1", newline="") as f:
        rows = list(csv.reader(f))[1:]
    return [normalize_text(row[0]) for row in rows if row and row[0].strip()]


def has_listed_word(text: str, lexicon: list[str]) -> bool:
    padded = f" {text} "  # the spaces make an entry match whole words only
    return any(f" {entry} " in padded for entry in lexicon)


def clean_examples(settings: dict, split: str, lexicon: list[str], rng: random.Random) -> list[str]:
    """A random sample of sentiment texts without a listed abusive word.

    The split files are sorted by label, so the sample must be random and never the first rows.
    """
    texts = []
    for period in settings["periods"]:
        rows = read_tsv(Path(settings["periods_dir"]) / period / f"{split}.tsv")
        usable = [text for text, _ in rows if not has_listed_word(text, lexicon)]
        texts += rng.sample(usable, min(len(usable), settings[f"{split}_per_period"]))
    return texts


def flag_rates(model: Pipeline, settings: dict, lexicon: list[str]) -> dict:
    """How often the detector flags the sentiment test sets. These texts have no abusive label.

    Some of them really are rude, so the rate is not an error rate. `has_listed_word` shows how
    many contain a word of the authors' list, which is roughly the share that may be rude.
    """
    rates = {}
    for period in settings["periods"]:
        rows = read_tsv(Path(settings["periods_dir"]) / period / "test.tsv")
        texts = [text for text, _ in rows]
        flagged = model.predict(texts).astype(bool)
        positive = np.array([label == "positive" for _, label in rows])
        rates[period] = {
            "flagged": float(flagged.mean()),
            "flagged_positive": float(flagged[positive].mean()),
            "has_listed_word": float(np.mean([has_listed_word(t, lexicon) for t in texts])),
        }
    return rates


def train(config: dict) -> dict:
    data_dir = Path(config["data_dir"])
    train_x, train_y = load(data_dir, "train")
    val_x, val_y = load(data_dir, "val")
    test_x, test_y = load(data_dir, "test")

    extra = config.get("clean_examples")
    lexicon = load_lexicon(Path(config["lexicon_path"])) if extra else []
    tune_x, tune_y = list(val_x), list(val_y)
    if extra:
        rng = random.Random(config["seed"])
        clean_train = clean_examples(extra, "train", lexicon, rng)
        clean_val = clean_examples(extra, "val", lexicon, rng)
        train_x, train_y = train_x + clean_train, train_y + [0] * len(clean_train)
        tune_x, tune_y = tune_x + clean_val, tune_y + [0] * len(clean_val)

    tune_scores = {}
    for c in config["logistic_regression"]["c_grid"]:
        tune_scores[c] = score(build(config, c).fit(train_x, train_y), tune_x, tune_y)
    best_c = max(tune_scores, key=lambda c: tune_scores[c]["f1_abusive"])

    # The chosen model is fit on the train data only. Val and test never train anything.
    model = build(config, best_c).fit(train_x, train_y)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")

    result = {
        "best_c": best_c,
        "tuning_by_c": {str(c): s for c, s in tune_scores.items()},
        "val_abusive_data": score(model, val_x, val_y),
        "test_abusive_data": score(model, test_x, test_y),
        "rows": {"train": len(train_x), "tuning": len(tune_x), "test": len(test_x)},
    }
    if extra:
        result["sentiment_test_flag_rates"] = flag_rates(model, extra, lexicon)
    results_path = Path(config["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/abusive.yaml"))
    args = parser.parse_args()

    result = train(yaml.safe_load(args.config.read_text()))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
