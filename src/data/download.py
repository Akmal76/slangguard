"""Download the raw datasets listed in a data config file."""

import argparse
import shutil
import urllib.request
from pathlib import Path

import yaml

TIMEOUT_SECONDS = 60


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file first, so a failed download never leaves a half file.
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response, tmp.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(dest)


def download_all(config: dict) -> list[Path]:
    raw_dir = Path(config["raw_dir"])
    saved = []
    for dataset, files in config["sources"].items():
        for filename, url in files.items():
            dest = raw_dir / dataset / filename
            download_file(url, dest)
            saved.append(dest)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    for path in download_all(config):
        print(f"saved {path}")


if __name__ == "__main__":
    main()
