import urllib.error

import pytest

from src.data.download import download_all, download_file


def test_download_all_saves_each_file_under_its_dataset(tmp_path):
    source = tmp_path / "source.tsv"
    source.write_text("text\tlabel\n")
    config = {
        "raw_dir": str(tmp_path / "raw"),
        "sources": {
            "smsa": {"train.tsv": source.as_uri()},
            "idsa": {"tweets.tsv": source.as_uri()},
        },
    }

    saved = download_all(config)

    assert saved == [tmp_path / "raw/smsa/train.tsv", tmp_path / "raw/idsa/tweets.tsv"]
    assert all(path.read_text() == "text\tlabel\n" for path in saved)


def test_failed_download_leaves_no_file(tmp_path):
    dest = tmp_path / "raw" / "missing.tsv"

    with pytest.raises(urllib.error.URLError):
        download_file((tmp_path / "does-not-exist.tsv").as_uri(), dest)

    assert not dest.exists()
