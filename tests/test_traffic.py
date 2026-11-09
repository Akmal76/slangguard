import pytest

from src.serving.traffic import is_local, load_texts, run


@pytest.mark.parametrize(
    "url", ["http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000"]
)
def test_local_addresses_are_allowed(url):
    assert is_local(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com",
        "https://api.example.com:8000",
        "http://10.0.0.5:8000",
        "http://localhost.evil.com",
    ],
)
def test_other_addresses_are_refused(url):
    assert not is_local(url)


@pytest.fixture
def periods_dir(tmp_path):
    path = tmp_path / "p2"
    path.mkdir()
    rows = ["text\tlabel"] + [f"text number {i}\tpositive" for i in range(50)]
    (path / "val.tsv").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_load_texts_returns_the_requested_number_and_is_repeatable(periods_dir):
    first = load_texts(periods_dir, "p2", "val", 20, seed=1)

    assert len(first) == 20
    assert first == load_texts(periods_dir, "p2", "val", 20, seed=1)
    assert first != load_texts(periods_dir, "p2", "val", 20, seed=2)
    assert set(first) <= {f"text number {i}" for i in range(50)}


def test_run_sends_every_text_at_the_given_rate_and_counts_the_successes():
    sent, pauses = [], []
    statuses = iter([200, 200, 422, 200])

    ok = run(
        ["a", "b", "c", "d"],
        "http://localhost:8000",
        rate=4,
        send_one=lambda url, text: (sent.append(text), next(statuses))[1],
        sleep=pauses.append,
    )

    assert sent == ["a", "b", "c", "d"]
    assert ok == 3
    assert pauses == [0.25] * 4
