import pytest

from src.evaluation.metrics import avg_f1, forgetting, macro_f1


def test_macro_f1_matches_hand_calculation():
    y_true = ["positive", "positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "negative", "neutral"]

    # negative: precision 1/2, recall 1 -> 2/3. neutral: 1.
    # positive: precision 1, recall 1/2 -> 2/3.
    assert macro_f1(y_true, y_pred) == pytest.approx((2 / 3 + 1 + 2 / 3) / 3)


def test_macro_f1_is_one_for_perfect_predictions():
    assert macro_f1(["neutral", "negative"], ["neutral", "negative"]) == 1.0


def test_macro_f1_counts_a_wrongly_predicted_class():
    # "positive" is predicted once but never true, so its F1 is 0 and pulls the mean down.
    assert macro_f1(["neutral", "neutral"], ["neutral", "positive"]) == pytest.approx(
        (2 / 3 + 0) / 2
    )


def test_avg_f1_is_the_mean_of_the_last_row():
    assert avg_f1([[0.9, 0.5], [0.6, 0.8]]) == pytest.approx(0.7)


def test_forgetting_with_two_steps():
    assert forgetting([[0.9, 0.5], [0.6, 0.8]]) == pytest.approx(0.3)


def test_forgetting_uses_the_best_earlier_score_of_each_period():
    a = [
        [0.9, 0.5, 0.4],
        [0.8, 0.85, 0.5],
        [0.7, 0.8, 0.9],
    ]

    # P1: best 0.9, final 0.7 -> 0.2. P2: best 0.85, final 0.8 -> 0.05. Mean 0.125.
    assert forgetting(a) == pytest.approx(0.125)


def test_forgetting_is_negative_when_old_periods_improve():
    assert forgetting([[0.5, 0.4], [0.7, 0.8]]) == pytest.approx(-0.2)


def test_forgetting_rejects_a_single_step():
    with pytest.raises(ValueError):
        forgetting([[0.9]])


def test_metrics_reject_a_matrix_that_is_not_square():
    with pytest.raises(ValueError):
        avg_f1([[0.9, 0.5]])
