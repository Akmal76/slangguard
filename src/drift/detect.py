"""Check whether new text has drifted away from the text the model has learned.

Evidently trains a small classifier to tell old text from new text and reports its ROC AUC.
An AUC near 0.5 means the two look the same. An AUC near 1 means the language has changed.
No labels are needed, so this works on data that has not been annotated yet.
"""

import random
from dataclasses import dataclass

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset


@dataclass(frozen=True)
class DriftResult:
    score: float  # ROC AUC of the old-versus-new classifier
    significant: bool  # Evidently: the classifier beats chance with 95% confidence
    drifted: bool  # significant, and the score is above our threshold
    snapshot: object  # the Evidently result, for the HTML report


def sample_texts(texts: list[str], max_rows: int | None, seed: int) -> list[str]:
    """A repeatable random sample, so one big period does not slow the check down."""
    if max_rows is None or max_rows >= len(texts):
        return texts
    return random.Random(seed).sample(texts, max_rows)


def _dataset(texts: list[str]) -> Dataset:
    return Dataset.from_pandas(
        pd.DataFrame({"text": texts}), data_definition=DataDefinition(text_columns=["text"])
    )


def check_drift(reference: list[str], current: list[str], threshold: float) -> DriftResult:
    """Compare new text (current) with the text the model learned (reference).

    Evidently's own flag only says the difference is not luck, and with enough texts even a
    tiny difference passes that test. So a period counts as drifted only when the score is also
    above `threshold`, which is set from controls in configs/pipeline.yaml.
    """
    snapshot = Report([DataDriftPreset()]).run(_dataset(current), _dataset(reference))
    metrics = {m["metric_name"].split("(")[0]: m["value"] for m in snapshot.dict()["metrics"]}
    score = float(metrics["ValueDrift"])
    significant = metrics["DriftedColumnsCount"]["count"] >= 1
    return DriftResult(score, significant, significant and score > threshold, snapshot)


def save_report(result: DriftResult, path) -> None:
    """Write Evidently's report as one HTML file (about 3.5 MB, mostly Evidently's own script)."""
    result.snapshot.save_html(str(path))
