import mlflow
import pytest
from mlflow import MlflowClient

from src.pipelines import registry
from src.pipelines.promote import Decision

MODEL = "test-model"
PERIODS = ["p1", "p2"]


@pytest.fixture(autouse=True)
def local_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    # Without this, run files would be written to ./mlruns in the repository.
    mlflow.create_experiment("test-exp", artifact_location=(tmp_path / "artifacts").as_uri())


@pytest.fixture
def adapter_dir(tmp_path):
    folder = tmp_path / "adapter"
    folder.mkdir()
    (folder / "adapter_config.json").write_text("{}")
    (folder / "adapter_model.safetensors").write_bytes(b"weights")
    return folder


def metrics(period="p2", champion=True):
    scores = {"val": [0.9, 0.7], "test": [0.8, 0.6]}
    return {
        "period": period,
        "periods": PERIODS,
        "best_val_f1": 0.7,
        "train_seconds": 12.0,
        "champion": scores if champion else None,
        "candidate": {"val": [0.89, 0.75], "test": [0.8, 0.65]},
    }


def log(adapter_dir, promote=True, **kwargs):
    decision = Decision(promote, "because")
    return registry.log_candidate("test-exp", metrics(**kwargs), decision, adapter_dir)


def test_log_candidate_records_scores_decision_and_adapter(adapter_dir):
    run_id = log(adapter_dir)

    run = mlflow.get_run(run_id)
    assert run.data.metrics["candidate_val_p2"] == pytest.approx(0.75)
    assert run.data.metrics["champion_test_p1"] == pytest.approx(0.8)
    assert run.data.tags["decision"] == "promote"
    assert run.data.tags["decision_reason"] == "because"
    files = {f.path for f in MlflowClient().list_artifacts(run_id, "adapter")}
    assert files == {"adapter/adapter_config.json", "adapter/adapter_model.safetensors"}


def test_first_candidate_has_no_champion_scores(adapter_dir):
    run = mlflow.get_run(log(adapter_dir, champion=False))

    assert not any(key.startswith("champion_") for key in run.data.metrics)


def test_there_is_no_champion_before_the_first_promotion():
    assert registry.get_champion(MODEL) is None


def test_a_registered_version_becomes_champion_only_after_the_alias_moves(adapter_dir):
    first = registry.register_version(MODEL, log(adapter_dir), "p2")
    assert registry.get_champion(MODEL) is None  # registered, not promoted

    registry.set_champion(MODEL, first)

    champion = registry.get_champion(MODEL)
    assert champion["version"] == first
    assert champion["tags"]["period"] == "p2"


def test_promoting_a_new_version_moves_the_alias_and_rollback_moves_it_back(adapter_dir):
    first = registry.register_version(MODEL, log(adapter_dir), "p2")
    registry.set_champion(MODEL, first)
    second = registry.register_version(MODEL, log(adapter_dir), "p3")

    registry.set_champion(MODEL, second)
    assert registry.get_champion(MODEL)["version"] == second

    registry.set_champion(MODEL, first)  # the rollback
    assert registry.get_champion(MODEL)["version"] == first


def test_tags_such_as_the_hub_revision_are_kept(adapter_dir):
    version = registry.register_version(MODEL, log(adapter_dir), "p2")
    registry.tag_version(MODEL, version, hf_repo="user/repo", hf_revision="abc123")
    registry.set_champion(MODEL, version)

    tags = registry.get_champion(MODEL)["tags"]

    assert (tags["hf_repo"], tags["hf_revision"]) == ("user/repo", "abc123")


def test_champion_adapter_can_be_downloaded(adapter_dir, tmp_path):
    version = registry.register_version(MODEL, log(adapter_dir), "p2")
    registry.set_champion(MODEL, version)

    folder = registry.download_adapter(registry.get_champion(MODEL), tmp_path / "download")

    assert (folder / "adapter_config.json").exists()
    assert (folder / "adapter_model.safetensors").read_bytes() == b"weights"


def test_a_refused_candidate_is_logged_but_not_registered(adapter_dir):
    log(adapter_dir, promote=False)

    assert MlflowClient().search_registered_models() == []
    assert len(mlflow.search_runs(experiment_names=["test-exp"])) == 1
