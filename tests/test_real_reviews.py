from pathlib import Path

from src.data.real_reviews import prepare, read_reviews, training_texts

HEADER = "Customer Rating,Customer Review,Sentiment\n"


def write_config(tmp_path: Path, reviews: str, train_rows: list[str]) -> dict:
    raw = tmp_path / "raw" / "real_reviews"
    raw.mkdir(parents=True)
    (raw / "prdect_id.csv").write_text(HEADER + reviews, encoding="utf-8")
    for name in ("train", "val", "test"):
        split_dir = tmp_path / "periods" / "p1"
        split_dir.mkdir(parents=True, exist_ok=True)
        rows = train_rows if name != "test" else ["text kasar\tnegative"]
        (split_dir / f"{name}.tsv").write_text("text\tlabel\n" + "\n".join(rows) + "\n")
    return {
        "raw_dir": str(tmp_path / "raw"),
        "periods_dir": str(tmp_path / "periods"),
        "real_reviews_dir": str(tmp_path / "out"),
    }


def test_read_reviews_cleans_text_and_maps_labels(tmp_path):
    config = write_config(tmp_path, '5,"Bagus Banget!!",Positive\n1,jelek,Negative\n', [])
    rows = read_reviews(Path(config["raw_dir"]) / "real_reviews" / "prdect_id.csv")
    assert rows == [("bagus banget", "positive", "5"), ("jelek", "negative", "1")]


def test_training_texts_skips_the_test_split(tmp_path):
    config = write_config(tmp_path, "", ["barang bagus\tpositive"])
    assert training_texts(Path(config["periods_dir"])) == {"barang bagus"}


def test_prepare_drops_duplicates_conflicts_and_texts_seen_in_training(tmp_path):
    reviews = (
        "5,barang bagus,Positive\n"  # in a training split: dropped
        "5,pengiriman cepat,Positive\n"
        "4,pengiriman cepat,Positive\n"  # same text and label: one copy
        "1,biasa saja,Negative\n"
        "5,biasa saja,Positive\n"  # same text, other label: dropped
        "2,kemasan rusak,Negative\n"
    )
    config = write_config(tmp_path, reviews, ["barang bagus\tpositive"])

    counts = prepare(config)

    lines = (tmp_path / "out" / "reviews.tsv").read_text().splitlines()
    assert counts == {"raw": 6, "kept": 2}
    assert lines[0] == "text\tlabel\trating"
    assert {line.split("\t")[0] for line in lines[1:]} == {"pengiriman cepat", "kemasan rusak"}
