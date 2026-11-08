"""Metrics for continual learning, as defined in docs/design.md section 8.

The result matrix `a` is a list of rows. `a[t][i]` is the macro-F1 on the test set of
period i after the model was trained up to step t (both counted from 0).
"""

Matrix = list[list[float]]


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    """Mean F1 over the classes that appear in the true or predicted labels."""
    classes = set(y_true) | set(y_pred)
    scores = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred, strict=True))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred, strict=True))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred, strict=True))
        # F1 = 2tp / (2tp + fp + fn). It is 0 when the class is never predicted correctly.
        scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def _check_square(a: Matrix) -> int:
    if not a or any(len(row) != len(a) for row in a):
        raise ValueError(
            "The result matrix must be square: one row per step, one column per period"
        )
    return len(a)


def avg_f1(a: Matrix) -> float:
    """Mean F1 over all periods after the last step."""
    _check_square(a)
    return sum(a[-1]) / len(a[-1])


def forgetting(a: Matrix) -> float:
    """Mean drop on old periods from their best earlier score to the final score.

    Lower is better. The best score is taken over all steps before the last one, as in
    the formula in design.md. A negative value means the old periods got better.
    """
    n = _check_square(a)
    if n < 2:
        raise ValueError("Forgetting needs at least 2 steps")
    drops = [max(a[t][i] for t in range(n - 1)) - a[-1][i] for i in range(n - 1)]
    return sum(drops) / len(drops)
