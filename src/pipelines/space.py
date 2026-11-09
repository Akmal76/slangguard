"""Build, publish and update the demo Space on Hugging Face.

The demo is a static web page. The model runs in the visitor's browser, so the Space needs no
server and costs nothing (Gradio and Docker Spaces on the free CPU need a paid plan).
The model is the champion, exported by src/pipelines/browser_model.py.
"""

import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi

# The files of the page, copied from web/. The model folder is added next to them.
PAGE_FILES = ["README.md", "index.html", "style.css", "app.js", "normalize.js", "examples.json"]
MODEL_DIR = "model"
FAILED_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}


def assemble(root: Path, model_dir: Path, out: Path) -> list[str]:
    """Copy the page and the exported model into `out`. Only these files go: no data."""
    for name in PAGE_FILES:
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / "web" / name, out / name)
    shutil.copytree(model_dir, out / MODEL_DIR, dirs_exist_ok=True)
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())


def publish(api: HfApi, space: str, folder: Path) -> None:
    """Create the static Space if needed and upload the page and the model."""
    api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=space,
        repo_type="space",
        commit_message="Update the demo from the project",
    )


def update_model(api: HfApi, space: str, model_dir: Path, revision: str) -> None:
    """Put a new champion on the page. A static Space needs no restart."""
    api.upload_folder(
        folder_path=str(model_dir),
        path_in_repo=MODEL_DIR,
        repo_id=space,
        repo_type="space",
        commit_message=f"Serve champion revision {revision[:8]}",
    )


def wait_until_running(
    api: HfApi, space: str, timeout_seconds: float = 300, poll_seconds: float = 10
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        stage = api.get_space_runtime(space).stage
        if stage == "RUNNING":
            return
        if stage in FAILED_STAGES:
            raise RuntimeError(f"The Space {space} is in stage {stage}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"The Space {space} is still {stage} after waiting")
        time.sleep(poll_seconds)
