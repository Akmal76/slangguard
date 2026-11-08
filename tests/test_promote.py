import copy

from src.pipelines.promote import decide

PERIODS = ["p1", "p2", "p3", "p4"]
RULE = {"min_new_gain": 0.0, "max_old_drop": 0.02}


def metrics(champion_val, candidate_val, period="p3"):
    return {
        "period": period,
        "periods": PERIODS,
        "champion": {"val": champion_val, "test": [0.5] * 4},
        "candidate": {"val": candidate_val, "test": [0.5] * 4},
    }


def test_better_on_the_new_period_without_forgetting_is_promoted():
    decision = decide(metrics([0.90, 0.70, 0.40, 0.3], [0.89, 0.71, 0.68, 0.3]), **RULE)

    assert decision.promote
    assert "better on p3" in decision.reason


def test_better_on_the_new_period_but_forgetting_an_old_one_is_refused():
    decision = decide(metrics([0.90, 0.70, 0.40, 0.3], [0.86, 0.71, 0.80, 0.3]), **RULE)

    assert not decision.promote
    assert "forgets p1" in decision.reason


def test_not_better_on_the_new_period_is_refused():
    decision = decide(metrics([0.90, 0.70, 0.60, 0.3], [0.91, 0.72, 0.60, 0.3]), **RULE)

    assert not decision.promote
    assert "not better on p3" in decision.reason


def test_a_drop_of_exactly_the_limit_is_allowed():
    decision = decide(metrics([0.90, 0.70, 0.40, 0.3], [0.88, 0.70, 0.70, 0.3]), **RULE)

    assert decision.promote


def test_an_older_period_that_improves_does_not_matter():
    decision = decide(metrics([0.90, 0.70, 0.40, 0.3], [0.95, 0.80, 0.70, 0.3]), **RULE)

    assert decision.promote


def test_the_first_model_is_promoted_because_there_is_no_champion():
    first = metrics([0.0] * 4, [0.9, 0.5, 0.4, 0.3], period="p1")
    first["champion"] = None

    assert decide(first, **RULE).promote


def test_periods_after_the_new_one_are_ignored():
    # p4 has not been learned yet. Its scores can be anything.
    decision = decide(metrics([0.90, 0.70, 0.40, 0.9], [0.90, 0.70, 0.70, 0.1]), **RULE)

    assert decision.promote


def test_test_scores_never_change_the_decision():
    base = metrics([0.90, 0.70, 0.40, 0.3], [0.89, 0.71, 0.68, 0.3])
    changed = copy.deepcopy(base)
    changed["candidate"]["test"] = [0.0, 0.0, 0.0, 0.0]
    changed["champion"]["test"] = [1.0, 1.0, 1.0, 1.0]

    assert decide(base, **RULE) == decide(changed, **RULE)


def test_a_required_gain_can_be_set():
    decision = decide(
        metrics([0.90, 0.70, 0.60, 0.3], [0.90, 0.70, 0.62, 0.3]),
        min_new_gain=0.05,
        max_old_drop=0.02,
    )

    assert not decision.promote
