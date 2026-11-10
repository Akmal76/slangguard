"""Load the abusive language detector and score texts with it.

It is a scikit-learn pipeline (TF-IDF, then logistic regression) trained by
src/training/abusive.py. It is separate from the sentiment model: a text can be neutral in
sentiment and still contain abusive words.
"""

from dataclasses import dataclass

import joblib
from huggingface_hub import hf_hub_download
from sklearn.pipeline import Pipeline

from src.data.clean import normalize_text

INFO_KEYS = ("abusive_repo", "abusive_revision")


@dataclass(frozen=True)
class AbusiveResult:
    abusive: bool
    score: float  # probability that the text is abusive


class AbusiveDetector:
    def __init__(self, pipeline: Pipeline, info: dict, threshold: float):
        self.pipeline = pipeline
        self.info = info
        self.threshold = threshold

    @classmethod
    def load(cls, config: dict) -> "AbusiveDetector":
        # The revision is a fixed commit. This file is a pickle, and loading a pickle runs code,
        # so it must only ever come from a revision of our own repo that we have checked.
        path = hf_hub_download(
            config["abusive_repo"], config["abusive_file"], revision=config["abusive_revision"]
        )
        info = {key: config[key] for key in INFO_KEYS}
        return cls(joblib.load(path), info, config["abusive_threshold"])

    def predict(self, texts: list[str]) -> list[AbusiveResult]:
        # The detector was trained on normalized text, so it must be given the same.
        scores = self.pipeline.predict_proba([normalize_text(text) for text in texts])[:, 1]
        return [AbusiveResult(bool(score >= self.threshold), float(score)) for score in scores]
