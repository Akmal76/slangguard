import pytest

from src.training.data import (
    LABEL_TO_ID,
    METHODS,
    period_test_rows,
    replay_rows,
    train_rows,
    val_rows,
)


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
    return {
        "periods_dir": str(tmp_path),
        "periods": ["p1", "p2", "p3"],
        "max_rows_per_split": None,
        "seed": 42,
    }


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


def test_max_rows_per_split_takes_a_repeatable_sample_from_each_split(config):
    config["max_rows_per_split"] = 1

    rows = train_rows(config, "full", 1)

    assert len(rows) == 2
    assert rows[0][0].startswith("p1 train")
    assert rows[1][0].startswith("p2 train")
    assert rows == train_rows(config, "full", 1)


def test_unknown_method_is_rejected(config):
    with pytest.raises(ValueError):
        train_rows(config, "ewc", 0)


def test_label_ids_follow_the_label_order():
    assert LABEL_TO_ID == {"negative": 0, "neutral": 1, "positive": 2}


@pytest.fixture
def big_config(tmp_path):
    """Three periods with 20 rows in each split, so replay fractions give whole numbers."""
    for period in ("p1", "p2", "p3"):
        (tmp_path / period).mkdir()
        for split in ("train", "val", "test"):
            lines = ["text\tlabel"] + [f"{period} {split} {i}\tpositive" for i in range(20)]
            (tmp_path / period / f"{split}.tsv").write_text("\n".join(lines) + "\n")
    return {
        "periods_dir": str(tmp_path),
        "periods": ["p1", "p2", "p3"],
        "max_rows_per_split": None,
        "seed": 42,
        "replay_fraction": 0.25,
    }


def texts(rows):
    return [text for text, _ in rows]


def test_replay_buffer_has_the_fraction_of_every_old_period(big_config):
    buffer = texts(replay_rows(big_config, 2))

    assert sum(t.startswith("p1 train") for t in buffer) == 5
    assert sum(t.startswith("p2 train") for t in buffer) == 5
    assert len(buffer) == 10


def test_replay_buffer_is_empty_at_the_first_step(big_config):
    assert replay_rows(big_config, 0) == []


def test_replay_buffer_never_contains_val_test_or_current_rows(big_config):
    buffer = texts(replay_rows(big_config, 1))

    assert all(t.startswith("p1 train") for t in buffer)


def test_replay_buffer_of_an_old_period_is_the_same_at_every_step(big_config):
    at_step_1 = [t for t in texts(replay_rows(big_config, 1)) if t.startswith("p1")]
    at_step_2 = [t for t in texts(replay_rows(big_config, 2)) if t.startswith("p1")]

    assert at_step_1 == at_step_2


def test_replay_buffer_depends_on_the_seed(big_config):
    first = replay_rows(big_config, 2)

    assert first == replay_rows(big_config, 2)
    assert first != replay_rows({**big_config, "seed": 7}, 2)


def test_replay_buffer_keeps_at_least_one_row_for_a_tiny_fraction(big_config):
    big_config["replay_fraction"] = 0.001

    assert len(replay_rows(big_config, 1)) == 1


def test_replay_fraction_zero_gives_an_empty_buffer(big_config):
    big_config["replay_fraction"] = 0

    assert replay_rows(big_config, 2) == []
    assert texts(train_rows(big_config, "replay", 2)) == texts(train_rows(big_config, "naive", 2))


@pytest.mark.parametrize("method", ["replay", "lora_replay"])
def test_replay_methods_train_on_the_buffer_plus_the_new_period(big_config, method):
    rows = texts(train_rows(big_config, method, 2))

    assert len(rows) == 10 + 20
    assert sum(t.startswith("p3 train") for t in rows) == 20


def test_lora_trains_on_the_new_period_only(big_config):
    assert texts(train_rows(big_config, "lora", 1)) == texts(train_rows(big_config, "naive", 1))


@pytest.mark.parametrize("method", ["naive", "replay", "lora", "lora_replay"])
def test_all_methods_except_full_validate_on_the_current_period_only(big_config, method):
    assert {t.split()[0] for t in texts(val_rows(big_config, method, 2))} == {"p3"}


def test_full_retrain_validates_on_all_periods_so_far(big_config):
    assert {t.split()[0] for t in texts(val_rows(big_config, "full", 1))} == {"p1", "p2"}


def test_method_flags_match_the_design():
    assert (METHODS["lora_replay"].lora, METHODS["lora_replay"].replay) == (True, "buffer")
    assert METHODS["full"].fresh_model and not METHODS["naive"].fresh_model
