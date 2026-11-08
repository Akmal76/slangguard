import random

from src.data.split import (
    build_periods,
    load_slang_forms,
    most_frequent_words,
    rewrite,
    split_into_parts,
    split_train_val_test,
)
from src.data.tsv import write_tsv

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def make_rows(n_per_label: int) -> list[tuple[str, str]]:
    labels = ("negative", "neutral", "positive")
    return [(f"{label} text {i}", label) for label in labels for i in range(n_per_label)]


def test_load_slang_forms_keeps_single_words_sorted(tmp_path):
    path = tmp_path / "lexicon.csv"
    path.write_text(
        "slang,formal\nyg,yang\nyng,yang\ngpp,tidak apa-apa\nyang,yang\nudah,sudah\n",
        encoding="utf-8",
    )

    assert load_slang_forms(path) == {"yang": ["yg", "yng"], "sudah": ["udah"]}


def test_split_into_parts_has_no_shared_rows_and_same_label_mix():
    rows = make_rows(30)

    parts = split_into_parts(rows, 3, random.Random(0))

    assert sorted(row for part in parts for row in part) == sorted(rows)
    for part in parts:
        assert [label for _, label in part].count("neutral") == 10


def test_split_train_val_test_uses_ratios_per_label():
    splits = split_train_val_test(make_rows(20), RATIOS, random.Random(0))

    assert {name: len(rows) for name, rows in splits.items()} == {
        "train": 42,
        "val": 9,
        "test": 9,
    }
    all_texts = [text for rows in splits.values() for text, _ in rows]
    assert len(all_texts) == len(set(all_texts))


def test_most_frequent_words_skips_short_words_and_palindromes():
    rows = [("di kakak makan makan nasi", "neutral"), ("di makan nasi", "neutral")]

    assert most_frequent_words(rows, count=2) == {"makan", "nasi"}


def test_rewrite_with_rate_zero_changes_nothing():
    assert rewrite("sudah makan", {"sudah": ["udah"]}, 0.0, random.Random(0)) == "sudah makan"


def test_rewrite_with_rate_one_replaces_every_known_word():
    assert rewrite("sudah makan", {"sudah": ["udah"]}, 1.0, random.Random(0)) == "udah makan"


def test_rewrite_always_reverses_chosen_words():
    result = rewrite("ayo makan", {}, 0.0, random.Random(0), frozenset({"makan"}))

    assert result == "ayo nakam"


def test_build_periods_is_reproducible(tmp_path):
    interim = tmp_path / "interim"
    write_tsv(make_rows(20), interim / "smsa.tsv")
    write_tsv([(f"sudah {t}", y) for t, y in make_rows(30)], interim / "idsa.tsv")
    lexicon = tmp_path / "lexicon.csv"
    lexicon.write_text("slang,formal\nudah,sudah\n", encoding="utf-8")
    config = {
        "seed": 42,
        "interim_dir": str(interim),
        "lexicon_path": str(lexicon),
        "split_ratios": RATIOS,
        "slang_rate": {"p3": 0.3, "p4": 0.7},
        "reversed_word_count": 1,
    }

    first, second = build_periods(config), build_periods(config)

    assert first == second
    assert list(first) == ["p1", "p2", "p3", "p4"]
