import yaml

from src.data.abusive import clean_tweet, deduplicate, prepare, read_abusive

HEADER = (
    "Tweet,HS,Abusive,HS_Individual,HS_Group,HS_Religion,HS_Race,HS_Physical,HS_Gender,HS_Other"
)


def write_source(raw_dir, tweets):
    path = raw_dir / "abusive" / "re_dataset.csv"
    path.parent.mkdir(parents=True)
    lines = [HEADER] + [f'"{tweet}",0,{label},0,0,0,0,0,0,0' for tweet, label in tweets]
    path.write_bytes("\n".join(lines).encode("latin-1"))
    return path


def test_clean_tweet_removes_the_placeholders_and_escapes_of_the_source():
    raw = r"RT USER USER Dasar BODOH \n http URL \xf0\x9f\x98 banget"

    assert clean_tweet(raw) == "dasar bodoh http banget"


def test_clean_tweet_keeps_words_that_only_contain_a_placeholder_name():
    assert clean_tweet("userland runtime curl") == "userland runtime curl"


def test_clean_tweet_drops_non_ascii_characters():
    assert clean_tweet("mantap â\u0080¦ sekali") == "mantap sekali"


def test_read_abusive_maps_the_labels_and_reads_latin_1(tmp_path):
    path = write_source(tmp_path, [("kamu é baik", "0"), ("dasar bodoh", "1")])

    assert read_abusive(path) == [("kamu baik", "not_abusive"), ("dasar bodoh", "abusive")]


def test_deduplicate_drops_conflicts_and_empty_texts():
    rows = [
        ("a", "abusive"),
        ("a", "abusive"),
        ("b", "abusive"),
        ("b", "not_abusive"),
        ("", "abusive"),
    ]

    assert deduplicate(rows) == [("a", "abusive")]


def test_prepare_writes_three_disjoint_splits_with_both_labels(tmp_path):
    tweets = [(f"kata{i} biasa", "0") for i in range(40)] + [
        (f"kata{i} bodoh", "1") for i in range(20)
    ]
    write_source(tmp_path / "raw", tweets)
    config = {
        "raw_dir": str(tmp_path / "raw"),
        "abusive_dir": str(tmp_path / "out"),
        "seed": 1,
        "split_ratios": yaml.safe_load("{train: 0.7, val: 0.15, test: 0.15}"),
    }

    counts = prepare(config)

    assert sum(counts.values()) == 60
    seen = {}
    for name in counts:
        lines = (tmp_path / "out" / f"{name}.tsv").read_text(encoding="utf-8").splitlines()[1:]
        seen[name] = {line.split("\t")[0] for line in lines}
        assert {line.split("\t")[1] for line in lines} == {"abusive", "not_abusive"}
    assert not (
        seen["train"] & seen["val"] or seen["train"] & seen["test"] or seen["val"] & seen["test"]
    )
