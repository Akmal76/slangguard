"""Send texts from one period to the local API, so the dashboard has something to show.

    python -m src.serving.traffic --period p2 --requests 300 --rate 6

Run it once with an early period and then with a late one, and the word pieces per word on
the dashboard rises, the way it would when new slang arrives. The texts come from the
validation split of data/periods. That data may not be shared, so this tool only talks to
this computer and refuses any other address.
"""

import argparse
import json
import random
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from src.data.tsv import read_tsv

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def is_local(url: str) -> bool:
    return urlparse(url).hostname in LOCAL_HOSTS


def load_texts(periods_dir: Path, period: str, split: str, count: int, seed: int) -> list[str]:
    """`count` texts of a period, picked at random but the same way for the same seed."""
    rows = read_tsv(periods_dir / period / f"{split}.tsv")
    texts = [text for text, _ in rows]
    return random.Random(seed).choices(texts, k=count)


def send(url: str, text: str) -> int:
    """Ask for one prediction and return the HTTP status."""
    request = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps({"texts": [text]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        response.read()
        return response.status


def run(texts: list[str], url: str, rate: float, send_one=send, sleep=time.sleep) -> int:
    """Send the texts one by one, `rate` per second. Returns how many got a 200 answer."""
    ok = 0
    for text in texts:
        ok += send_one(url, text) == 200
        sleep(1 / rate)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", required=True, choices=["p1", "p2", "p3", "p4"])
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--rate", type=float, default=6, help="requests per second")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--periods-dir", type=Path, default=Path("data/periods"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not is_local(args.url):
        raise SystemExit(f"Refusing {args.url}: the period texts must not leave this computer.")
    texts = load_texts(args.periods_dir, args.period, args.split, args.requests, args.seed)
    ok = run(texts, args.url, args.rate)
    print(f"{args.period}: {ok} of {len(texts)} requests answered with 200")


if __name__ == "__main__":
    main()
