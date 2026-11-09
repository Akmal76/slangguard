"""The retraining pipeline, as Prefect flows.

    uv run --env-file .env python -m src.pipelines.flow --bootstrap p3     # the first champion
    uv run --env-file .env python -m src.pipelines.flow --new-period p4    # an update
    uv run --env-file .env python -m src.pipelines.flow --deploy-only      # restart the API

An update runs these steps, and stops early when a step says so:
drift check -> retrain on a Kaggle GPU -> promotion rule -> publish -> switch the API.
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import mlflow
import yaml
from huggingface_hub import HfApi
from prefect import flow, get_run_logger, task

from src.drift.detect import check_drift, sample_texts, save_report
from src.pipelines import registry
from src.pipelines.browser_model import export_champion
from src.pipelines.deploy import deploy
from src.pipelines.kaggle_job import run_job
from src.pipelines.promote import Decision, decide
from src.pipelines.publish import upload_adapter
from src.pipelines.space import update_model
from src.serving.config import load_serving_config
from src.training.data import load_split


def load_configs() -> tuple[dict, dict]:
    train = yaml.safe_load(Path("configs/train.yaml").read_text())
    pipeline = yaml.safe_load(Path("configs/pipeline.yaml").read_text())
    return train, pipeline


def ensure_tracking() -> None:
    # The same local default as the training scripts. With a .env file it is DagsHub instead.
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")


def make_workdir(path: str | None) -> str:
    """The folder for the report and the downloaded results. A given path is created if needed."""
    if path is None:
        return tempfile.mkdtemp(prefix="slangguard-")
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def pushed_commit() -> str:
    """The commit the Kaggle job runs. Kaggle clones it from GitHub, so it has to be there."""
    subprocess.run(["git", "fetch", "-q", "origin"], check=True)
    if subprocess.run(["git", "diff", "--quiet", "HEAD"]).returncode != 0:
        raise RuntimeError(
            "There are uncommitted changes. The Kaggle job runs only committed code."
        )
    if (
        subprocess.run(["git", "merge-base", "--is-ancestor", "HEAD", "origin/main"]).returncode
        != 0
    ):
        raise RuntimeError("The current commit is not on GitHub yet. Push it first.")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


@task
def drift_task(train_cfg: dict, pipeline_cfg: dict, reference: str, new: str, workdir: str) -> dict:
    settings = pipeline_cfg["drift"]

    def texts(period: str, seed: int) -> list[str]:
        rows = load_split(train_cfg, period, "train")
        return sample_texts([text for text, _ in rows], settings["max_rows"], seed)

    result = check_drift(
        texts(reference, settings["seed"]), texts(new, settings["seed"] + 1), settings["threshold"]
    )
    html = Path(workdir) / f"drift_{new}.html"
    save_report(result, html)
    registry.log_drift_check(
        pipeline_cfg["mlflow"]["experiment"], new, result.score, result.significant, html
    )
    return {
        "drift_score": result.score,
        "significant": result.significant,
        "drifted": result.drifted,
    }


@task
def retrain_task(
    train_cfg: dict, pipeline_cfg: dict, periods: list[str], champion: dict | None, workdir: str
) -> str:
    kaggle = pipeline_cfg["kaggle"]
    spec = {
        "commit": pushed_commit(),
        "code_repo": kaggle["code_repo"],
        "periods": periods,
        "champion": champion,
    }
    return str(run_job(kaggle, Path(train_cfg["periods_dir"]), spec, Path(workdir)))


@task(retries=2, retry_delay_seconds=10)
def upload_task(adapter_dir: str, repo: str, message: str) -> str:
    return upload_adapter(Path(adapter_dir), repo, message)


@task
def deploy_task(repo: str, revision: str) -> None:
    deploy(repo, revision)


@task
def space_task(pipeline_cfg: dict, repo: str, revision: str) -> None:
    """Export the champion for the browser and put it on the demo page."""
    config = load_serving_config(
        Path("configs/serving.yaml"), {"ADAPTER_REPO": repo, "ADAPTER_REVISION": revision}
    )
    with tempfile.TemporaryDirectory() as temp:
        export_champion(config, Path(temp))
        update_model(HfApi(), pipeline_cfg["space"]["repo"], Path(temp), revision)


def publish_champion(pipeline_cfg: dict, metrics: dict, adapter_dir: Path, run_id: str) -> dict:
    """Make a candidate the champion: store it, register it, move the alias, switch API and demo."""
    hub = pipeline_cfg["huggingface"]["repo"]
    model_name = pipeline_cfg["mlflow"]["model_name"]
    revision = upload_task(str(adapter_dir), hub, f"champion after {metrics['period']}")
    version = registry.register_version(model_name, run_id, metrics["period"])
    registry.tag_version(model_name, version, hf_repo=hub, hf_revision=revision)
    registry.set_champion(model_name, version)
    try:
        deploy_task(hub, revision)
        space_task(pipeline_cfg, hub, revision)
    except Exception as error:
        raise RuntimeError(
            f"Version {version} is the champion in the registry, but the API or the demo was not "
            f"switched ({error}). Fix the cause and run the flow with --deploy-only."
        ) from error
    return {"version": version, "revision": revision}


@flow(name="retrain")
def retrain_flow(new_period: str, workdir: str | None = None) -> dict:
    """Check a new period for drift and, if it drifted, update the champion."""
    log = get_run_logger()
    ensure_tracking()
    train_cfg, pipeline_cfg = load_configs()
    periods = train_cfg["periods"]
    model_name = pipeline_cfg["mlflow"]["model_name"]
    experiment = pipeline_cfg["mlflow"]["experiment"]

    champion = registry.get_champion(model_name)
    if champion is None:
        raise RuntimeError("There is no champion yet. Run the flow with --bootstrap first.")
    learned = champion["tags"]["period"]
    if periods.index(new_period) != periods.index(learned) + 1:
        raise ValueError(
            f"The champion has learned up to {learned}, so the next period is not {new_period}"
        )
    work = make_workdir(workdir)

    drift = drift_task(train_cfg, pipeline_cfg, learned, new_period, work)
    log.info(f"drift score {drift['drift_score']:.3f}, drifted: {drift['drifted']}")
    if not drift["drifted"]:
        return {"status": "no drift", **drift}

    previous = {"repo": champion["tags"]["hf_repo"], "revision": champion["tags"]["hf_revision"]}
    results = Path(retrain_task(train_cfg, pipeline_cfg, [new_period], previous, work))
    metrics = json.loads((results / new_period / "metrics.json").read_text())
    decision = decide(metrics, **pipeline_cfg["promotion"])
    log.info(f"decision: {'promote' if decision.promote else 'refuse'}: {decision.reason}")

    adapter_dir = results / new_period / "adapter"
    run_id = registry.log_candidate(experiment, metrics, decision, adapter_dir)
    if not decision.promote:
        return {"status": "refused", "reason": decision.reason, **drift}
    outcome = publish_champion(pipeline_cfg, metrics, adapter_dir, run_id)
    return {"status": "promoted", "reason": decision.reason, **drift, **outcome}


@flow(name="bootstrap")
def bootstrap_flow(last_period: str = "p3", workdir: str | None = None) -> dict:
    """Make the first champion by training period after period up to `last_period`."""
    ensure_tracking()
    train_cfg, pipeline_cfg = load_configs()
    periods = train_cfg["periods"]
    model_name = pipeline_cfg["mlflow"]["model_name"]

    existing = registry.get_champion(model_name)
    if existing is not None:
        raise RuntimeError(f"There is already a champion (version {existing['version']}).")
    trained = periods[: periods.index(last_period) + 1]
    work = make_workdir(workdir)

    results = Path(retrain_task(train_cfg, pipeline_cfg, trained, None, work))
    metrics = json.loads((results / last_period / "metrics.json").read_text())
    decision = Decision(True, f"first champion, trained on {trained[0]} to {last_period}")
    adapter_dir = results / last_period / "adapter"
    run_id = registry.log_candidate(
        pipeline_cfg["mlflow"]["experiment"], metrics, decision, adapter_dir
    )
    outcome = publish_champion(pipeline_cfg, metrics, adapter_dir, run_id)
    return {"status": "promoted", "reason": decision.reason, **outcome}


@flow(name="deploy-champion")
def deploy_champion_flow() -> dict:
    """Switch the API and the demo to the champion in the registry, for example after a failure."""
    ensure_tracking()
    _, pipeline_cfg = load_configs()
    champion = registry.get_champion(pipeline_cfg["mlflow"]["model_name"])
    if champion is None:
        raise RuntimeError("There is no champion to deploy.")
    repo, revision = champion["tags"]["hf_repo"], champion["tags"]["hf_revision"]
    deploy_task(repo, revision)
    space_task(pipeline_cfg, repo, revision)
    return {"status": "deployed", "version": champion["version"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--new-period", help="check this period and update the champion if needed")
    what.add_argument("--bootstrap", metavar="LAST_PERIOD", help="make the first champion")
    what.add_argument("--deploy-only", action="store_true")
    args = parser.parse_args()

    if args.new_period:
        result = retrain_flow(args.new_period)
    elif args.bootstrap:
        result = bootstrap_flow(args.bootstrap)
    else:
        result = deploy_champion_flow()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
