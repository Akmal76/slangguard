import re

import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.model import SentimentModel

INFO = {"base_model": "tiny", "adapter_repo": "me/champions", "adapter_revision": "rev123"}
CONFIG = {"max_texts": 3, "max_chars": 40}


@pytest.fixture
def client(make_model, tokenizer):
    model = SentimentModel(make_model(), tokenizer, INFO, max_length=8)
    with TestClient(create_app(CONFIG, model)) as client:
        yield client


def scrape(client) -> str:
    return client.get("/metrics").text


def value(text: str, series: str) -> float:
    """The number after a series such as slangguard_requests_total{endpoint="/predict",...}."""
    found = re.search(rf"^{re.escape(series)} ([0-9.e+-]+)$", text, re.MULTILINE)
    assert found, f"{series} is not in the metrics"
    return float(found.group(1))


def test_metrics_are_served_in_the_prometheus_text_format(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE slangguard_requests_total counter" in response.text


def test_the_live_model_is_reported(client):
    text = scrape(client)

    series = 'slangguard_model_info{adapter_repo="me/champions",adapter_revision="rev123"}'
    assert value(text, series) == 1


def test_requests_are_counted_by_endpoint_and_status(client):
    client.post("/predict", json={"texts": ["enak banget"]})
    client.post("/predict", json={"texts": ["biasa", "parah banget"]})
    client.post("/predict", json={"texts": []})  # refused

    text = scrape(client)

    assert value(text, 'slangguard_requests_total{endpoint="/predict",status="200"}') == 2
    assert value(text, 'slangguard_requests_total{endpoint="/predict",status="422"}') == 1
    assert value(text, 'slangguard_request_seconds_count{endpoint="/predict"}') == 3


def test_every_predicted_text_is_counted_by_label_and_size(client):
    client.post("/predict", json={"texts": ["enak banget", "biasa", "parah banget"]})

    text = scrape(client)

    labels = [
        value(text, f'slangguard_predictions_total{{label="{label}"}}')
        for label in ("negative", "neutral", "positive")
        if f'label="{label}"' in text
    ]
    assert sum(labels) == 3
    assert value(text, "slangguard_text_words_count") == 3
    assert value(text, "slangguard_text_words_sum") == 5  # 2 + 1 + 2 words
    assert value(text, "slangguard_pieces_per_word_count") == 3


def test_pieces_per_word_is_one_when_every_word_is_known(client):
    client.post("/predict", json={"texts": ["enak banget"]})

    text = scrape(client)

    assert value(text, "slangguard_pieces_per_word_sum") == pytest.approx(1.0)


def test_unknown_paths_do_not_create_new_series(client):
    for path in ("/nope", "/also/nope", "/x/y/z"):
        client.get(path)

    text = scrape(client)

    assert value(text, 'slangguard_requests_total{endpoint="unmatched",status="404"}') == 3
    assert "/nope" not in text


def test_the_text_that_users_send_is_never_exported(client):
    client.post("/predict", json={"texts": ["rahasia banget enak"]})

    assert "rahasia" not in scrape(client)


def test_each_app_has_its_own_numbers(make_model, tokenizer):
    model = SentimentModel(make_model(), tokenizer, INFO, max_length=8)
    first_app, second_app = create_app(CONFIG, model), create_app(CONFIG, model)
    with TestClient(first_app) as first, TestClient(second_app) as second:
        first.post("/predict", json={"texts": ["enak banget"]})

        assert 'endpoint="/predict"' in first.get("/metrics").text
        assert 'endpoint="/predict"' not in second.get("/metrics").text
