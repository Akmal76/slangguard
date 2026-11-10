import joblib
import pytest
from fastapi.testclient import TestClient

from src.serving import abusive as abusive_module
from src.serving.abusive import AbusiveDetector
from src.serving.app import create_app
from src.serving.model import SentimentModel
from src.training.abusive import build

BAD = ["dasar bodoh", "kamu goblok", "anjing banget", "tolol sekali", "bangsat lu"]
GOOD = ["makanan enak sekali", "pelayanan ramah sekali", "tempat bersih nyaman", "harga murah"]
TRAIN_CONFIG = {
    "seed": 1,
    "tfidf": {"analyzer": "char_wb", "ngram_range": [2, 4], "min_df": 1, "sublinear_tf": True},
    "logistic_regression": {"class_weight": "balanced", "max_iter": 200},
}
INFO = {"base_model": "tiny", "adapter_repo": "none", "adapter_revision": "abc123"}
ABUSIVE_INFO = {"abusive_repo": "me/abusive", "abusive_revision": "r1"}


@pytest.fixture
def pipeline():
    return build(TRAIN_CONFIG, c=10.0).fit(BAD + GOOD, [1] * len(BAD) + [0] * len(GOOD))


@pytest.fixture
def detector(pipeline):
    return AbusiveDetector(pipeline, ABUSIVE_INFO, threshold=0.5)


def make_client(make_model, tokenizer, detector):
    model = SentimentModel(make_model(), tokenizer, INFO, max_length=8)
    config = {"max_texts": 5, "max_chars": 50}
    return TestClient(create_app(config, model, detector))


def test_the_detector_flags_abusive_words_and_not_clean_text(detector):
    results = detector.predict(["Goblok LU!", "makanan enak"])

    assert [r.abusive for r in results] == [True, False]
    assert results[0].score > 0.5 > results[1].score


def test_the_threshold_decides_the_flag(pipeline):
    strict = AbusiveDetector(pipeline, ABUSIVE_INFO, threshold=1.1)

    assert not strict.predict(["goblok lu"])[0].abusive


def test_load_reads_the_pinned_file_of_the_repo(pipeline, tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    joblib.dump(pipeline, path)
    asked = []

    def fake_download(repo, filename, revision):
        asked.append((repo, filename, revision))
        return str(path)

    monkeypatch.setattr(abusive_module, "hf_hub_download", fake_download)
    config = {**ABUSIVE_INFO, "abusive_file": "model.joblib", "abusive_threshold": 0.4}

    loaded = AbusiveDetector.load(config)

    assert asked == [("me/abusive", "model.joblib", "r1")]
    assert loaded.threshold == 0.4 and loaded.info == ABUSIVE_INFO
    assert loaded.predict(["goblok lu"])[0].abusive


def test_predict_adds_the_abusive_flag_next_to_the_sentiment(make_model, tokenizer, detector):
    with make_client(make_model, tokenizer, detector) as client:
        response = client.post("/predict", json={"texts": ["goblok lu", "makanan enak"]})

    first, second = response.json()["predictions"]
    assert (first["abusive"], second["abusive"]) == (True, False)
    assert first["abusive_score"] > 0.5 > second["abusive_score"]
    assert first["label"] in first["probabilities"]  # the sentiment is still there


def test_without_a_detector_the_abusive_fields_are_empty(make_model, tokenizer):
    with make_client(make_model, tokenizer, None) as client:
        body = client.post("/predict", json={"texts": ["goblok lu"]}).json()
        health = client.get("/health").json()

    assert body["predictions"][0]["abusive"] is None
    assert body["predictions"][0]["abusive_score"] is None
    assert "abusive_model" not in health


def test_health_names_the_abusive_model(make_model, tokenizer, detector):
    with make_client(make_model, tokenizer, detector) as client:
        assert client.get("/health").json()["abusive_model"] == ABUSIVE_INFO


def test_metrics_count_flagged_and_clean_texts(make_model, tokenizer, detector):
    with make_client(make_model, tokenizer, detector) as client:
        client.post("/predict", json={"texts": ["goblok lu", "bodoh amat", "makanan enak"]})
        text = client.get("/metrics").text

    assert 'slangguard_abusive_texts_total{abusive="true"} 2.0' in text
    assert 'slangguard_abusive_texts_total{abusive="false"} 1.0' in text
    assert "goblok" not in text  # only counts are exported, never the text
