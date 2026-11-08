"""Make the API serve a new model: write the setting, recreate the container, wait for it."""

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def write_champion_env(path: Path, repo: str, revision: str) -> None:
    """compose.yaml reads this file, and the API reads the two values at start."""
    path.write_text(f"ADAPTER_REPO={repo}\nADAPTER_REVISION={revision}\n")


def fetch_health(url: str) -> dict | None:
    """The answer of GET /health, or None while the API is not up."""
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=5) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def wait_for_revision(
    url: str,
    revision: str,
    timeout_seconds: float = 300,
    poll_seconds: float = 3,
    fetch=fetch_health,
) -> None:
    """Wait until the API reports the given adapter revision."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        health = fetch(url)
        if health and health["model"]["adapter_revision"] == revision:
            return
        if time.monotonic() > deadline:
            seen = health["model"]["adapter_revision"] if health else "no answer"
            raise TimeoutError(
                f"The API did not serve revision {revision} in time (last seen: {seen})"
            )
        time.sleep(poll_seconds)


def deploy(
    repo: str,
    revision: str,
    env_file: Path = Path("champion.env"),
    url: str = "http://localhost:8000",
    run=subprocess.run,
    fetch=fetch_health,
) -> None:
    """Switch the API to this model. It is down for a few seconds while the container restarts.

    The same call is the rollback: give it the repo and revision of the earlier champion.
    """
    write_champion_env(env_file, repo, revision)
    # --build makes sure the image has the current code. It is fast when nothing changed.
    run(["docker", "compose", "up", "-d", "--build", "--force-recreate", "api"], check=True)
    wait_for_revision(url, revision, fetch=fetch)
