import random

from src.drift.detect import check_drift, sample_texts, save_report

OLD_WORDS = ["makanan", "pelayanan", "tempat", "harga", "rasa", "enak", "bagus", "ramah", "mahal"]
NEW_WORDS = ["wkwk", "gaje", "mantul", "santuy", "baper", "kuy", "gabut", "mager", "bucin"]


def make_texts(words, n=400, seed=0):
    rng = random.Random(seed)
    return [" ".join(rng.choices(words, k=6)) for _ in range(n)]


def test_same_language_is_not_drift():
    result = check_drift(make_texts(OLD_WORDS, seed=1), make_texts(OLD_WORDS, seed=2), 0.6)

    assert result.score < 0.6
    assert not result.drifted


def test_new_vocabulary_is_drift():
    result = check_drift(make_texts(OLD_WORDS), make_texts(NEW_WORDS), 0.6)

    assert result.score > 0.9
    assert result.significant
    assert result.drifted


def test_a_change_below_the_threshold_is_not_reported():
    # The difference is real (Evidently flags it), but the project threshold is stricter.
    result = check_drift(make_texts(OLD_WORDS), make_texts(NEW_WORDS), threshold=1.0)

    assert result.significant
    assert not result.drifted


def test_report_is_saved_as_html(tmp_path):
    result = check_drift(make_texts(OLD_WORDS), make_texts(NEW_WORDS), 0.6)

    save_report(result, tmp_path / "drift.html")

    assert "<html" in (tmp_path / "drift.html").read_text(encoding="utf-8")[:2000].lower()


def test_sample_texts_is_repeatable_and_keeps_small_inputs():
    texts = [f"text {i}" for i in range(100)]

    assert sample_texts(texts, 10, seed=1) == sample_texts(texts, 10, seed=1)
    assert sample_texts(texts, 10, seed=1) != sample_texts(texts, 10, seed=2)
    assert len(sample_texts(texts, 10, seed=1)) == 10
    assert sample_texts(texts, 500, seed=1) == texts
    assert sample_texts(texts, None, seed=1) == texts


def test_sample_texts_can_return_nothing():
    assert sample_texts(["a", "b"], 0, seed=1) == []
