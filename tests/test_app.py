import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.model import SentimentModel

INFO = {"base_model": "tiny", "adapter_repo": "none", "adapter_revision": "abc123"}
CONFIG = {"max_texts": 3, "max_chars": 20}


@pytest.fixture
def client(make_model, tokenizer):
    model = SentimentModel(make_model(), tokenizer, INFO, max_length=8)
    with TestClient(create_app(CONFIG, model)) as client:  # "with" runs the start-up code
        yield client


def predict(client, texts):
    return client.post("/predict", json={"texts": texts})


def test_health_reports_ok_and_the_model_version(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": INFO}


def test_predict_returns_one_prediction_per_text_in_order(client):
    response = predict(client, ["enak banget", "parah banget"])

    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    assert body["model"] == INFO
    for prediction in body["predictions"]:
        probabilities = prediction["probabilities"]
        assert list(probabilities) == ["negative", "neutral", "positive"]
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert prediction["label"] == max(probabilities, key=probabilities.get)


def test_predict_gives_the_same_result_for_the_same_text(client):
    first = predict(client, ["biasa"]).json()["predictions"]
    second = predict(client, ["Biasa!"]).json()["predictions"]  # same text after cleaning

    assert first == second


@pytest.mark.parametrize(
    "texts",
    [
        [],  # nothing to label
        ["a", "b", "c", "d"],  # more than max_texts
        ["x" * 21],  # longer than max_chars
        ["enak", "!!!"],  # no words after cleaning
        ["   "],
    ],
)
def test_predict_rejects_bad_requests(client, texts):
    assert predict(client, texts).status_code == 422


@pytest.mark.parametrize("body", [{}, {"texts": "enak"}, {"texts": [1, 2]}, {"text": ["enak"]}])
def test_predict_rejects_the_wrong_shape(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_the_error_says_which_text_is_the_problem(client):
    response = predict(client, ["enak", "!!!"])

    assert "Text 1" in response.json()["detail"]


def test_the_documentation_page_is_available(client):
    assert client.get("/docs").status_code == 200
