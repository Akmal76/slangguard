"""The script that runs on Kaggle. It trains update steps on the GPU and writes the results.

The pipeline copies this file, puts the job settings into SPEC, and pushes it as a Kaggle
kernel (see kaggle_job.py). It holds no secrets: the code comes from a public GitHub commit,
the champion from a public Hugging Face repo, and the data from a private Kaggle dataset.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

SPEC: dict = {}  # replaced when the job is pushed


def run(*command: str) -> None:
    print("+", *command, flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. The job was pushed without a GPU.")

    work = Path("/kaggle/working")
    repo = work / "repo"
    run("git", "clone", "-q", SPEC["code_repo"], str(repo))
    os.chdir(repo)
    run("git", "checkout", "-q", SPEC["commit"])

    # The Kaggle image has a torchao that peft refuses (checked in Week 3). It is not used here.
    run(sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao")
    run(sys.executable, "-m", "pip", "install", "-q", "peft==0.21.2", "pytest")
    # A check on a tiny random model, so a version clash fails now and not during training.
    run(sys.executable, "-m", "pytest", "-q", "tests/test_train.py", "-k", "lora")

    # The dataset has flat files named like p1_train.tsv. Put them where the code expects them.
    source = next(Path("/kaggle/input").rglob("p1_train.tsv")).parent
    for file in source.glob("p?_*.tsv"):
        period, split = file.stem.split("_")
        (repo / "data" / "periods" / period).mkdir(parents=True, exist_ok=True)
        shutil.copy(file, repo / "data" / "periods" / period / f"{split}.tsv")

    champion = None
    if SPEC["champion"]:
        from huggingface_hub import snapshot_download

        champion = snapshot_download(
            SPEC["champion"]["repo"], revision=SPEC["champion"]["revision"]
        )

    results = work / "result"
    for period in SPEC["periods"]:
        command = [sys.executable, "-m", "src.training.update", "--period", period]
        command += ["--config", "configs/train.yaml", "--out", str(results / period)]
        if champion:
            command += ["--champion", str(champion)]
        run(*command)
        champion = results / period / "adapter"  # the next step continues from this one

    os.chdir(work)
    shutil.rmtree(repo)  # the output then holds only the results, not the data


if __name__ == "__main__":
    main()
