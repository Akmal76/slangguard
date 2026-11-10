"""The numbers the API reports to Prometheus at GET /metrics.

Only counts and sizes are recorded. The text that users send is never stored or exported.
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class Metrics:
    def __init__(self) -> None:
        # Each API gets its own registry, so tests can build many apps without clashing.
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "slangguard_requests_total",
            "Requests by endpoint and HTTP status",
            ["endpoint", "status"],
            registry=self.registry,
        )
        self.request_seconds = Histogram(
            "slangguard_request_seconds",
            "Time to answer a request",
            ["endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2.5),
            registry=self.registry,
        )
        self.predictions = Counter(
            "slangguard_predictions_total",
            "Predicted labels. The label mix is a sign of drift.",
            ["label"],
            registry=self.registry,
        )
        self.abusive = Counter(
            "slangguard_abusive_texts_total",
            "Texts the abusive language detector flagged, or not.",
            ["abusive"],
            registry=self.registry,
        )
        self.words = Histogram(
            "slangguard_text_words",
            "Words per text, after cleaning",
            buckets=(1, 2, 3, 5, 8, 12, 20, 30, 50),
            registry=self.registry,
        )
        # The buckets follow what was measured on the period data: about 1.1 words for formal
        # reviews (P1) and 1.6 for the heaviest slang (P4).
        self.pieces_per_word = Histogram(
            "slangguard_pieces_per_word",
            "Word pieces per word. It rises when slang the vocabulary does not know arrives.",
            buckets=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2, 3),
            registry=self.registry,
        )
        self.model_info = Gauge(
            "slangguard_model_info",
            "The model that is live. The value is always 1.",
            ["adapter_repo", "adapter_revision"],
            registry=self.registry,
        )

    def record_predictions(self, results) -> None:
        for result in results:
            label = max(result.probabilities, key=result.probabilities.get)
            self.predictions.labels(label).inc()
            self.words.observe(result.words)
            self.pieces_per_word.observe(result.pieces / result.words)

    def record_abusive(self, results) -> None:
        for result in results:
            self.abusive.labels(str(result.abusive).lower()).inc()

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
