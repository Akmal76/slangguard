"""Judge the models on real shopper reviews (PRDECT-ID), by the rules in configs/real_reviews.yaml.

    uv run python -m src.evaluation.real_reviews

Run it once. The set is a test set: never tune on it.
"""

import argparse
import json
import random
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from src.data.clean import LABELS
from src.pipelines.browser_model import predict_onnx
from src.serving.abusive import AbusiveDetector
from src.serving.config import load_serving_config
from src.serving.model import SentimentModel

REAL_CLASSES = ("positive", "negative")


def read_reviews(path: Path) -> list[tuple[str, str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()[1:]
    return [tuple(line.split("\t")) for line in lines]


def two_class_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Macro-F1 over positive and negative. A neutral prediction is a miss, never a class."""
    scores = []
    for c in REAL_CLASSES:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred, strict=True))
        # A class that is neither true nor predicted anywhere has no F1. It can only happen in a
        # tiny resample, and it is skipped like in metrics.macro_f1.
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else 0.0


def bootstrap_interval(y_true, y_pred, resamples: int, seed: int) -> tuple[float, float]:
    """95% interval of the score: score again on reviews drawn with replacement."""
    rng = random.Random(seed)
    n = len(y_true)
    scores = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        scores.append(two_class_f1([y_true[i] for i in idx], [y_pred[i] for i in idx]))
    scores.sort()
    return scores[int(0.025 * resamples)], scores[int(0.975 * resamples)]


def batches(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def predict_labels(predict_probabilities, texts: list[str], batch_size: int) -> list[str]:
    """Labels for all texts. Short texts go together, so batches carry little padding."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    labels = [""] * len(texts)
    for chunk in batches(order, batch_size):
        for i, row in zip(chunk, predict_probabilities([texts[i] for i in chunk]), strict=True):
            labels[i] = LABELS[max(range(len(LABELS)), key=lambda k: row[k])]
    return labels


def summarize(rows, predicted: list[str], settings: dict) -> dict:
    gold = [label for _, label, _ in rows]
    keep = [i for i, (_, _, rating) in enumerate(rows) if rating != "3"]

    def score(indices):
        y, p = [gold[i] for i in indices], [predicted[i] for i in indices]
        low, high = bootstrap_interval(y, p, settings["bootstrap_resamples"], settings["seed"])
        return {"f1": two_class_f1(y, p), "ci95": [low, high], "reviews": len(indices)}

    neutral = sum(p == "neutral" for p in predicted)
    return {
        "all_reviews": score(range(len(rows))),
        "without_3_stars": score(keep),
        "neutral_share": neutral / len(predicted),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/real_reviews.yaml"))
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())

    rows = read_reviews(Path(settings["data_path"]))
    texts = [text for text, _, _ in rows]
    size = settings["batch_size"]
    api_config = load_serving_config(Path(settings["champion_config"]), {})
    browser_dir = Path(settings["browser_model_dir"])
    browser_info = json.loads((browser_dir / "model-info.json").read_text())

    results = {"reviews": len(rows), "targets": settings["targets"]}

    # 1. The model people use: the 8-bit champion that the browser demo loads.
    tokenizer = AutoTokenizer.from_pretrained(browser_dir)
    onnx_path = browser_dir / "onnx" / "model_quantized.onnx"
    int8 = predict_labels(lambda b: predict_onnx(onnx_path, tokenizer, b), texts, size)
    results["browser_int8"] = summarize(rows, int8, settings) | {"model": browser_info}

    # 2. The same champion in full precision, and 3. the model that the API serves.
    champion_config = api_config | {
        k: browser_info[k] for k in ("adapter_repo", "adapter_revision")
    }
    for name, config in (("champion_float", champion_config), ("api_model", api_config)):
        model = SentimentModel.load(config)
        labels = predict_labels(
            lambda b, m=model: [[p[label] for label in LABELS] for p in m.predict(b)], texts, size
        )
        results[name] = summarize(rows, labels, settings) | {"model": model.info}

    # The abusive flag on the same reviews. These reviews have no abusive label: it is a rate.
    detector = AbusiveDetector.load(api_config)
    flags = [r.abusive for batch in batches(texts, size) for r in detector.predict(batch)]
    gold = [label for _, label, _ in rows]
    positives = [f for f, g in zip(flags, gold, strict=True) if g == "positive"]
    results["abusive_flag"] = {
        "all": sum(flags) / len(flags),
        "positive_reviews": sum(positives) / len(positives),
    }

    Path(settings["results_path"]).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
