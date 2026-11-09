"""Put a champion adapter on Hugging Face, where the Kaggle job and the API download it."""

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
license: cc-by-sa-4.0
library_name: peft
base_model: indobenchmark/indobert-base-p1
---

# SlangGuard champion adapter

The LoRA adapter that the SlangGuard retraining pipeline currently serves. Each commit in this
repository is one champion, so the revision identifies the model. The pipeline replaces the
files when a new model passes the promotion rule. For the released model with a full model
card, see https://huggingface.co/amc76/slangguard-indobert-lora-replay and the project at
https://github.com/Akmal76/slangguard.
"""


def upload_adapter(adapter_dir: Path, repo: str, message: str) -> str:
    """Upload the adapter as a new commit and return the commit hash (the revision)."""
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp) / "adapter"
        shutil.copytree(adapter_dir, folder)
        (folder / "README.md").write_text(CARD)  # replaces the empty template that PEFT writes
        commit = api.upload_folder(
            folder_path=str(folder), repo_id=repo, repo_type="model", commit_message=message
        )
    return commit.oid
