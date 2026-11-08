import pytest

from src.training.data import LABEL_TO_ID, period_test_rows, train_rows, val_rows


@pytest.fixture
def config(tmp_path):
    for period in ("p1", "p2", "p3"):
        (tmp_path / period).mkdir()
        for split in ("train", "val", "test"):
            lines = [
                "text\tlabel",
                f"{period} {split} a\tpositive",
                f"{period} {split} b\tnegative",
            ]
            (tmp_path / period / f"{split}.tsv").write_text("\n".join(lines) + "\n")
    return {"periods_dir": str(tmp_path), "periods": ["p1", "p2", "p3"], "max_rows_per_split": None}


def test_naive_trains_only_on_the_current_period(config):
    texts = [text for text, _ in train_rows(config, "naive", 1)]

    assert texts == ["p2 train a", "p2 train b"]


def test_full_trains_on_all_periods_so_far(config):
    texts = [text for text, _ in train_rows(config, "full", 1)]

    assert texts == ["p1 train a", "p1 train b", "p2 train a", "p2 train b"]


def test_validation_follows_the_same_periods_as_training(config):
    naive = {text for text, _ in val_rows(config, "naive", 2)}
    full = {text for text, _ in val_rows(config, "full", 1)}

    assert naive == {"p3 val a", "p3 val b"}
    assert full == {"p1 val a", "p1 val b", "p2 val a", "p2 val b"}


def test_validation_never_includes_a_future_period(config):
    texts = [text for text, _ in val_rows(config, "full", 0)]

    assert texts == ["p1 val a", "p1 val b"]


def test_test_rows_come_from_the_test_split_of_one_period(config):
    assert [text for text, _ in period_test_rows(config, "p3")] == ["p3 test a", "p3 test b"]


def test_max_rows_per_split_keeps_the_first_rows(config):
    config["max_rows_per_split"] = 1

    assert train_rows(config, "full", 1) == [("p1 train a", "positive"), ("p2 train a", "positive")]


def test_unknown_method_is_rejected(config):
    with pytest.raises(ValueError):
        train_rows(config, "lora", 0)


def test_label_ids_follow_the_label_order():
    assert LABEL_TO_ID == {"negative": 0, "neutral": 1, "positive": 2}
