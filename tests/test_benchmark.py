import pytest

from src.serving.benchmark import summarize


def test_summarize_reports_median_and_95th_percentile():
    stats = summarize([float(i) for i in range(1, 101)])  # 1 ms ... 100 ms

    assert stats["p50"] == pytest.approx(50.5)
    assert stats["p95"] == pytest.approx(95.05)
    assert stats["max"] == 100.0
