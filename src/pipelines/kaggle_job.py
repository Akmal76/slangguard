"""Run an update step on a Kaggle GPU from the laptop, using the Kaggle command line tool.

The data goes to a private Kaggle dataset, and the job is a script kernel (kaggle_body.py)
with the job settings written into it. Nothing secret is sent: the kernel needs no token.
"""

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

BODY = Path(__file__).with_name("kaggle_body.py")
MARKER = "SPEC: dict = {}  # replaced when the job is pushed"
PERIOD_FILES = ("train", "val", "test")


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run the Kaggle command line tool. It reads the account from ~/.kaggle."""
    executable = shutil.which("kaggle") or str(Path(sys.executable).parent / "kaggle")
    return subprocess.run([executable, *args], capture_output=True, text=True, check=check)


def render_script(spec: dict) -> str:
    """The job script with its settings filled in."""
    body = BODY.read_text()
    if MARKER not in body:
        raise ValueError("kaggle_body.py has lost the line that holds the job settings")
    return body.replace(MARKER, f"SPEC: dict = {spec!r}")


def prepare_dataset(periods_dir: Path, folder: Path, owner: str, slug: str) -> None:
    """Copy the period files flat into `folder`, with the metadata Kaggle needs."""
    folder.mkdir(parents=True, exist_ok=True)
    for period_dir in sorted(periods_dir.glob("p?")):
        for split in PERIOD_FILES:
            shutil.copy(period_dir / f"{split}.tsv", folder / f"{period_dir.name}_{split}.tsv")
    metadata = {
        "title": "slangguard periods",
        "id": f"{owner}/{slug}",
        "licenses": [{"name": "other"}],
    }
    (folder / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))


def upload_dataset(folder: Path, owner: str, slug: str, message: str, wait: float = 5) -> None:
    """Create the private dataset, or add a version to it, and wait until it is ready."""
    ref = f"{owner}/{slug}"
    if kaggle("datasets", "status", ref, check=False).returncode == 0:
        kaggle("datasets", "version", "-p", str(folder), "-m", message)
    else:
        kaggle("datasets", "create", "-p", str(folder))  # datasets are private by default
    for _ in range(60):
        if "ready" in kaggle("datasets", "status", ref, check=False).stdout.lower():
            return
        time.sleep(wait)
    raise TimeoutError(f"Kaggle dataset {ref} is not ready after waiting")


def kernel_metadata(cfg: dict) -> dict:
    return {
        "id": f"{cfg['owner']}/{cfg['kernel']}",
        "title": cfg["kernel"],
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,  # for GitHub and Hugging Face
        "dataset_sources": [f"{cfg['owner']}/{cfg['dataset']}"],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "docker_image": cfg["docker_image"],
        "machine_shape": "NvidiaTeslaT4",
    }


def push_kernel(cfg: dict, spec: dict, folder: Path) -> None:
    """Push the job script. Kaggle starts running it right away, on a GPU."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "script.py").write_text(render_script(spec))
    (folder / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(cfg), indent=2))
    kaggle("kernels", "push", "-p", str(folder))


def wait_for_kernel(cfg: dict) -> None:
    """Wait until the job ends. Raise if it failed or took too long."""
    ref = f"{cfg['owner']}/{cfg['kernel']}"
    deadline = time.monotonic() + cfg["timeout_minutes"] * 60
    while True:
        found = re.search(r"KernelWorkerStatus\.(\w+)", kaggle("kernels", "status", ref).stdout)
        status = found.group(1) if found else "UNKNOWN"
        if status == "COMPLETE":
            return
        if status in ("ERROR", "CANCEL_ACKNOWLEDGED", "CANCEL_REQUESTED"):
            raise RuntimeError(f"Kaggle job {ref} ended with status {status}")
        if time.monotonic() > deadline:
            # The Kaggle tool cannot stop a job. The script ends when it finishes or fails.
            raise TimeoutError(f"Kaggle job {ref} is still {status}. Check it on kaggle.com.")
        time.sleep(cfg["poll_seconds"])


def fetch_results(cfg: dict, destination: Path) -> Path:
    """Download what the job wrote. Returns the folder with one subfolder per period."""
    destination.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{cfg['owner']}/{cfg['kernel']}", "-p", str(destination))
    return destination / "result"


def run_job(cfg: dict, periods_dir: Path, spec: dict, workdir: Path) -> Path:
    """Upload the data, run the job and download the results."""
    prepare_dataset(periods_dir, workdir / "dataset", cfg["owner"], cfg["dataset"])
    upload_dataset(
        workdir / "dataset", cfg["owner"], cfg["dataset"], f"periods for {spec['periods']}"
    )
    push_kernel(cfg, spec, workdir / "kernel")
    wait_for_kernel(cfg)
    return fetch_results(cfg, workdir / "output")
