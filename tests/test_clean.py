import pytest

from src.data.clean import clean_all, deduplicate, drop_malay, normalize_text, read_idsa


def test_normalize_text_lowercases_and_collapses_spaces():
    assert normalize_text("  Mantap   BANGETTT \t ") == "mantap bangettt"


def test_normalize_text_replaces_punctuation_with_spaces():
    assert normalize_text("enak . murah , teman-teman!") == "enak murah teman teman"


def test_normalize_text_joins_apostrophes_and_keeps_emoticon_tokens():
    assert normalize_text("don't __laugh__") == "dont __laugh__"


def test_normalize_text_removes_standalone_amp_only():
    assert normalize_text("makan amp minum &amp; camp ampas") == "makan minum camp ampas"


def test_drop_malay_removes_rows_with_a_marker_word():
    rows = [("aku nak makan", "neutral"), ("aku mau makan", "neutral")]

    assert drop_malay(rows, {"nak"}) == [("aku mau makan", "neutral")]


def test_drop_malay_matches_whole_words_only():
    assert drop_malay([("anak kucing", "positive")], {"nak"}) == [("anak kucing", "positive")]


def test_deduplicate_keeps_one_copy():
    rows = [("enak", "positive"), ("enak", "positive"), ("biasa", "neutral")]

    assert deduplicate(rows) == [("enak", "positive"), ("biasa", "neutral")]


def test_deduplicate_drops_texts_with_conflicting_labels():
    rows = [("parah", "positive"), ("parah", "negative"), ("biasa", "neutral")]

    assert deduplicate(rows) == [("biasa", "neutral")]


def test_deduplicate_drops_empty_text():
    assert deduplicate([("", "neutral")]) == []


def test_deduplicate_rejects_unknown_label():
    with pytest.raises(ValueError):
        deduplicate([("enak", "pos")])


def test_read_idsa_maps_numeric_labels(tmp_path):
    (tmp_path / "idsa").mkdir()
    (tmp_path / "idsa" / "tweets.tsv").write_text("sentimen\tTweet\n-1\tsebel\n0\tbiasa\n1\tasik\n")

    assert read_idsa(tmp_path) == [
        ("sebel", "negative"),
        ("biasa", "neutral"),
        ("asik", "positive"),
    ]


def test_read_idsa_rejects_unknown_label(tmp_path):
    (tmp_path / "idsa").mkdir()
    (tmp_path / "idsa" / "tweets.tsv").write_text("sentimen\tTweet\n2\taneh\n")

    with pytest.raises(ValueError):
        read_idsa(tmp_path)


def test_clean_all_removes_idsa_texts_that_appear_in_smsa(tmp_path):
    raw = tmp_path / "raw"
    (raw / "smsa").mkdir(parents=True)
    (raw / "idsa").mkdir()
    (raw / "smsa" / "train.tsv").write_text("Makanan Enak\tpositive\n")
    (raw / "smsa" / "valid.tsv").write_text("pelayanan lambat\tnegative\n")
    (raw / "smsa" / "test.tsv").write_text("")
    (raw / "idsa" / "tweets.tsv").write_text(
        "sentimen\tTweet\n1\tmakanan  enak\n0\tlagi di rumah\n0\tnak makan\n"
    )
    config = {
        "raw_dir": str(raw),
        "interim_dir": str(tmp_path / "interim"),
        "malay_markers": ["nak"],
    }

    counts = clean_all(config)

    idsa_output = (tmp_path / "interim" / "idsa.tsv").read_text()
    assert counts == {"smsa": 2, "idsa": 1}
    assert idsa_output == "text\tlabel\nlagi di rumah\tneutral\n"
