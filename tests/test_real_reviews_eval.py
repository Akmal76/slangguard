import pytest

from src.evaluation.real_reviews import predict_labels, summarize, two_class_f1


def test_neutral_prediction_is_a_miss_not_a_class():
    gold = ["positive", "positive", "negative", "negative"]
    predicted = ["positive", "neutral", "negative", "negative"]
    # positive: tp 1, fn 1 -> 2/3. negative: tp 2 -> 1. Neutral has no score of its own.
    assert two_class_f1(gold, predicted) == pytest.approx((2 / 3 + 1) / 2)


def test_perfect_prediction_scores_one():
    assert two_class_f1(["positive", "negative"], ["positive", "negative"]) == 1.0


def test_summarize_can_leave_out_three_star_reviews():
    rows = [("a", "positive", "5"), ("b", "negative", "1"), ("c", "negative", "3")]
    settings = {"bootstrap_resamples": 10, "seed": 1}

    result = summarize(rows, ["positive", "negative", "positive"], settings)

    assert result["all_reviews"]["reviews"] == 3
    assert result["without_3_stars"]["reviews"] == 2
    assert result["without_3_stars"]["f1"] == 1.0
    assert result["all_reviews"]["f1"] < 1.0


def test_predict_labels_keeps_the_order_of_the_texts():
    texts = ["panjang sekali teksnya", "ok", "sedang saja"]
    answers = {"panjang sekali teksnya": [0, 0, 1], "ok": [1, 0, 0], "sedang saja": [0, 1, 0]}

    labels = predict_labels(lambda batch: [answers[t] for t in batch], texts, batch_size=2)

    assert labels == ["positive", "negative", "neutral"]
