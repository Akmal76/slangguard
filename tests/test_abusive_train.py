import json
import random

import joblib

from src.data.tsv import write_tsv
from src.training.abusive import (
    clean_examples,
    has_listed_word,
    load,
    load_lexicon,
    train,
)

BAD = ["dasar bodoh", "kamu goblok", "anjing banget", "tolol sekali", "bangsat lu"]
GOOD = ["makanan enak sekali", "pelayanan ramah sekali", "tempat bersih nyaman", "harga murah"]


def make_config(tmp_path):
    for split in ("train", "val", "test"):
        rows = [(f"{text} {i}", "abusive") for i, text in enumerate(BAD)]
        rows += [(f"{text} {i}", "not_abusive") for i, text in enumerate(GOOD)]
        write_tsv(rows, tmp_path / "data" / f"{split}.tsv")
    return {
        "data_dir": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "out"),
        "results_path": str(tmp_path / "results" / "abusive.json"),
        "seed": 1,
        "tfidf": {"analyzer": "char_wb", "ngram_range": [2, 4], "min_df": 1, "sublinear_tf": True},
        "logistic_regression": {"c_grid": [1.0, 10.0], "class_weight": "balanced", "max_iter": 200},
    }


def test_load_turns_labels_into_zero_and_one(tmp_path):
    make_config(tmp_path)

    texts, labels = load(tmp_path / "data", "train")

    assert len(texts) == len(labels) == 9
    assert sum(labels) == 5


def test_train_saves_a_model_that_finds_the_abusive_words(tmp_path):
    config = make_config(tmp_path)

    result = train(config)

    model = joblib.load(tmp_path / "out" / "model.joblib")
    assert list(model.predict(["goblok lu", "makanan enak"])) == [1, 0]
    assert result["best_c"] in (1.0, 10.0)
    assert set(result["test_abusive_data"]) == {
        "f1_abusive",
        "precision_abusive",
        "recall_abusive",
        "macro_f1",
    }


def test_train_writes_the_scores_next_to_the_choice_of_c(tmp_path):
    config = make_config(tmp_path)

    train(config)

    saved = json.loads((tmp_path / "results" / "abusive.json").read_text())
    assert set(saved["tuning_by_c"]) == {"1.0", "10.0"}
    assert saved["rows"] == {"train": 9, "tuning": 9, "test": 9}


def write_periods(tmp_path):
    for period in ("p1", "p2"):
        rows = [(f"makanan enak sekali nomor {i} {period}", "positive") for i in range(30)]
        rows += [(f"dasar goblok nomor {i} {period}", "negative") for i in range(5)]
        for split in ("train", "val", "test"):
            write_tsv(rows, tmp_path / "periods" / period / f"{split}.tsv")


def write_lexicon(tmp_path):
    path = tmp_path / "abusive.csv"
    path.write_bytes(b"ABUSIVE\ngoblok\nbacot banget\n")
    return path


def test_load_lexicon_cleans_each_entry_like_the_texts(tmp_path):
    path = tmp_path / "abusive.csv"
    path.write_bytes(b"ABUSIVE\nGoblok!\n\nbacot  banget\n")

    assert load_lexicon(path) == ["goblok", "bacot banget"]


def test_has_listed_word_matches_whole_words_and_phrases_only():
    lexicon = ["goblok", "bacot banget"]

    assert has_listed_word("kamu goblok sekali", lexicon)
    assert has_listed_word("ih bacot banget sih", lexicon)
    assert not has_listed_word("goblokan itu", lexicon)
    assert not has_listed_word("bacot saja", lexicon)


def test_clean_examples_skip_rude_texts_and_cap_each_period(tmp_path):
    write_periods(tmp_path)
    settings = {
        "periods_dir": str(tmp_path / "periods"),
        "periods": ["p1", "p2"],
        "val_per_period": 10,
    }

    texts = clean_examples(settings, "val", ["goblok"], random.Random(1))

    assert len(texts) == 20
    assert not any("goblok" in text for text in texts)
    assert texts == clean_examples(settings, "val", ["goblok"], random.Random(1))  # the seed rules


def test_train_adds_clean_examples_and_reports_the_flag_rate_on_sentiment_tests(tmp_path):
    config = make_config(tmp_path)
    write_periods(tmp_path)
    config["lexicon_path"] = str(write_lexicon(tmp_path))
    config["clean_examples"] = {
        "periods_dir": str(tmp_path / "periods"),
        "periods": ["p1", "p2"],
        "train_per_period": 10,
        "val_per_period": 5,
    }

    result = train(config)

    assert result["rows"] == {"train": 9 + 20, "tuning": 9 + 10, "test": 9}
    rates = result["sentiment_test_flag_rates"]
    assert set(rates) == {"p1", "p2"}
    assert rates["p1"]["flagged"] < 0.5  # mostly clean praise
    assert rates["p1"]["has_listed_word"] == 5 / 35
