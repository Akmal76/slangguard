"""Build, publish and update the demo Space on Hugging Face.

The Space is a separate repository. It gets a copy of the code that the API runs, so the demo
and the API always give the same answer for the same model.
"""

import shutil
import time
from pathlib import Path

from huggingface_hub import HfApi

# Copied unchanged from the project, next to the files of demo/.
SOURCES = [
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/tsv.py",
    "src/data/clean.py",
    "src/serving/__init__.py",
    "src/serving/config.py",
    "src/serving/model.py",
    "configs/serving.yaml",
]
DEMO_FILES = ["app.py", "README.md", "requirements.txt"]
FAILED_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}


def assemble(root: Path, out: Path) -> list[str]:
    """Copy the Space files into `out`. Only these files go: no data, no notebooks."""
    names = [*DEMO_FILES, *SOURCES]
    for name in names:
        source = root / "demo" / name if name in DEMO_FILES else root / name
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, destination)
    return names


def publish(api: HfApi, space: str, folder: Path, repo: str, revision: str) -> None:
    """Create the Space if needed, say which model it should load, and upload the files."""
    api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
    # The variables come first, so the first build already loads the right model.
    api.add_space_variable(space, "ADAPTER_REPO", repo)
    api.add_space_variable(space, "ADAPTER_REVISION", revision)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=space,
        repo_type="space",
        commit_message="Update the demo from the project",
    )


def point_at_champion(api: HfApi, space: str, repo: str, revision: str) -> None:
    """Make the Space load another model. Called by the pipeline when it promotes a champion."""
    api.add_space_variable(space, "ADAPTER_REPO", repo)
    api.add_space_variable(space, "ADAPTER_REVISION", revision)
    api.restart_space(space)


def wait_until_running(
    api: HfApi, space: str, timeout_seconds: float = 900, poll_seconds: float = 15
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
