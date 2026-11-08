"""Keep track of candidate models and the champion in MLflow.

Every candidate is logged as a run, so the history of decisions is kept. Only candidates that
pass the promotion rule become a registered model version, and the alias `champion` points to
the version that is live now. Moving that alias is the promotion, and moving it back is the
rollback.
"""

from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.pipelines.promote import Decision

CHAMPION = "champion"


def log_candidate(experiment: str, metrics: dict, decision: Decision, adapter_dir: Path) -> str:
    """Log one update step as a run and return the run id."""
    mlflow.set_experiment(experiment)
    periods = metrics["periods"]
    with mlflow.start_run(run_name=f"update-{metrics['period']}") as run:
        mlflow.log_param("period", metrics["period"])
        for who in ("champion", "candidate"):
            if metrics[who] is None:
                continue
            for split in ("val", "test"):
                for period, f1 in zip(periods, metrics[who][split], strict=True):
                    mlflow.log_metric(f"{who}_{split}_{period}", f1)
        mlflow.log_metrics(
            {"best_val_f1": metrics["best_val_f1"], "train_seconds": metrics["train_seconds"]}
        )
        mlflow.set_tags(
            {
                "decision": "promote" if decision.promote else "refuse",
                "decision_reason": decision.reason,
            }
        )
        mlflow.log_artifacts(str(adapter_dir), "adapter")
        mlflow.log_dict(metrics, "metrics.json")
        return run.info.run_id


def register_version(model_name: str, run_id: str, period: str) -> str:
    """Register the adapter of a run as a new model version and return its number.

    The adapter is a folder of plain files, not an MLflow model, so the version is created
    straight from the folder's location. (`mlflow.register_model` expects a logged model.)
    """
    client = MlflowClient()
    try:
        client.create_registered_model(model_name)
    except MlflowException as error:
        if error.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
    source = f"{mlflow.get_run(run_id).info.artifact_uri}/adapter"
    return client.create_model_version(model_name, source, run_id, tags={"period": period}).version


def tag_version(model_name: str, version: str, **tags: str) -> None:
    client = MlflowClient()
    for key, value in tags.items():
        client.set_model_version_tag(model_name, version, key, value)


def set_champion(model_name: str, version: str) -> None:
    MlflowClient().set_registered_model_alias(model_name, CHAMPION, version)


def get_champion(model_name: str) -> dict | None:
    """The version the alias `champion` points to, or None if there is none yet."""
    try:
        version = MlflowClient().get_model_version_by_alias(model_name, CHAMPION)
    except MlflowException as error:
        # Two cases mean "no champion yet": the model does not exist (RESOURCE_DOES_NOT_EXIST),
        # or it exists but the alias is not set (reported as INVALID_PARAMETER_VALUE).
        message = str(error).lower()
        if error.error_code == "RESOURCE_DOES_NOT_EXIST" or (
            "alias" in message and "not found" in message
        ):
            return None
        raise  # a wrong password or a server error must not look like "no champion"
    return {
        "version": version.version,
        "run_id": version.run_id,
        "source": version.source,
        "tags": dict(version.tags),
    }


def download_adapter(champion: dict, destination: Path) -> Path:
    """Copy the adapter files of a registered version to a local folder."""
    path = mlflow.artifacts.download_artifacts(
        artifact_uri=champion["source"], dst_path=str(destination)
    )
    return Path(path)
