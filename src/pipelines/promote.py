"""The rule that decides whether a new model replaces the champion.

The rule looks only at validation scores. Test sets are scored and reported, but they never
decide anything, so they stay an honest measure of the model that was chosen.
"""

from dataclasses import dataclass

TOLERANCE = 1e-9  # so a drop of exactly the limit is not refused because of rounding


@dataclass(frozen=True)
class Decision:
    promote: bool
    reason: str


def decide(metrics: dict, min_new_gain: float, max_old_drop: float) -> Decision:
    """Compare the candidate with the champion, using the metrics from src/training/update.py.

    The candidate is promoted when it is better on the new period and has not forgotten the
    periods learned before. Periods that come after the new one are not looked at.
    """
    if metrics["champion"] is None:
        return Decision(True, "there is no champion yet")

    periods = metrics["periods"]
    t = periods.index(metrics["period"])
    champion, candidate = metrics["champion"]["val"], metrics["candidate"]["val"]

    gain = candidate[t] - champion[t]
    if gain <= min_new_gain + TOLERANCE:
        return Decision(
            False,
            f"not better on {periods[t]}: val F1 {candidate[t]:.3f} against {champion[t]:.3f} "
            f"(needs a gain above {min_new_gain:.3f})",
        )

    drops = {periods[i]: champion[i] - candidate[i] for i in range(t)}
    for period, drop in drops.items():
        if drop > max_old_drop + TOLERANCE:
            return Decision(
                False,
                f"forgets {period}: val F1 falls by {drop:.3f}, above the limit "
                f"{max_old_drop:.3f} (it gains {gain:.3f} on {periods[t]})",
            )

    worst = f"{max(drops.values()):+.3f}" if drops else "none"
    return Decision(
        True,
        f"better on {periods[t]} by {gain:.3f}; the biggest drop on an older period is {worst}",
    )
