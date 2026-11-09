"""The flows run for real (Prefect, Evidently, the promotion rule and a local MLflow registry).
Only what costs money or leaves the machine is replaced: Kaggle, Hugging Face and Docker."""

import json
import random

import mlflow
import pytest
from prefect.testing.utilities import prefect_test_harness

from src.pipelines import flow as flow_module
from src.pipelines import registry
from src.pipelines.promote import Decision

OLD_WORDS = ["makanan", "pelayanan", "tempat", "harga", "rasa", "enak", "bagus", "ramah", "mahal"]
NEW_WORDS = ["wkwk", "gaje", "mantul", "santuy", "baper", "kuy", "gabut", "mager", "bucin"]
PERIODS = ["p1", "p2", "p3", "p4"]

PIPELINE = {
    "drift": {"threshold": 0.6, "max_rows": 300, "seed": 1},
    "promotion": {"min_new_gain": 0.0, "max_old_drop": 0.02},
    "mlflow": {"experiment": "flow-exp", "model_name": "flow-model"},
    "kaggle": {"code_repo": "https://example.com/repo.git"},
    "huggingface": {"repo": "me/champions"},
    "space": {"repo": "me/demo"},
}
GOOD = {"champion": [0.90, 0.80, 0.50, 0.3], "candidate": [0.89, 0.80, 0.80, 0.5]}
FORGETS = {"champion": [0.90, 0.80, 0.50, 0.3], "candidate": [0.70, 0.80, 0.85, 0.5]}


@pytest.fixture(scope="module", autouse=True)
def prefect_runtime():
    with prefect_test_harness():
        yield


def make_texts(words, seed, n=300):
    rng = random.Random(seed)
    return [" ".join(rng.choices(words, k=6)) for _ in range(n)]


class World:
    """The fakes, and a record of what the flow did with them."""

    def __init__(self, tmp_path, monkeypatch, p4_is_new_language=True):
        periods_dir = tmp_path / "periods"
        for i, period in enumerate(PERIODS):
            words = NEW_WORDS if period == "p4" and p4_is_new_language else OLD_WORDS
            for j, split in enumerate(("train", "val", "test")):
                path = periods_dir / period / f"{split}.tsv"
                path.parent.mkdir(parents=True, exist_ok=True)
                rows = [f"{text}\tpositive" for text in make_texts(words, 10 * i + j)]
                path.write_text("\n".join(["text\tlabel", *rows]) + "\n")
        self.train_cfg = {
            "periods": PERIODS,
            "periods_dir": str(periods_dir),
            "max_rows_per_split": None,
            "seed": 1,
        }
        self.tmp_path = tmp_path
        self.jobs, self.uploads, self.deploys = [], [], []
        self.exported, self.space_updates, self.space_error = [], [], None
        self.scores = GOOD
        self.deploy_error = None

        mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
        mlflow.create_experiment("flow-exp", artifact_location=(tmp_path / "artifacts").as_uri())
        monkeypatch.setattr(flow_module, "ensure_tracking", lambda: None)
        monkeypatch.setattr(flow_module, "load_configs", lambda: (self.train_cfg, PIPELINE))
        monkeypatch.setattr(flow_module, "pushed_commit", lambda: "abc123")
        monkeypatch.setattr(flow_module, "run_job", self.run_job)
        monkeypatch.setattr(flow_module, "upload_adapter", self.upload_adapter)
        monkeypatch.setattr(flow_module, "deploy", self.deploy)
        monkeypatch.setattr(flow_module, "export_champion", self.export_champion)
        monkeypatch.setattr(flow_module, "update_model", self.update_model)
        monkeypatch.setattr(flow_module, "HfApi", lambda: "api")

    def run_job(self, cfg, periods_dir, spec, workdir):
        self.jobs.append(spec)
        results = workdir / "output" / "result"
        for period in spec["periods"]:
            (results / period / "adapter").mkdir(parents=True)
            (results / period / "adapter" / "adapter_config.json").write_text("{}")
            metrics = {
                "period": period,
                "periods": PERIODS,
                "best_val_f1": 0.8,
                "train_seconds": 20.0,
                "champion": None
                if spec["champion"] is None and period == "p1"
                else {"val": self.scores["champion"], "test": [0.5] * 4},
                "candidate": {"val": self.scores["candidate"], "test": [0.5] * 4},
            }
            (results / period / "metrics.json").write_text(json.dumps(metrics))
        return results

    def upload_adapter(self, adapter_dir, repo, message):
        self.uploads.append((repo, message))
        return "rev-new"

    def deploy(self, repo, revision):
        if self.deploy_error:
            raise self.deploy_error
        self.deploys.append((repo, revision))

    def export_champion(self, config, out_dir):
        self.exported.append((config["adapter_repo"], config["adapter_revision"]))
        (out_dir / "model.onnx").write_text("exported")

    def update_model(self, api, space, model_dir, revision):
        if self.space_error:
            raise self.space_error
        self.space_updates.append((space, revision, sorted(p.name for p in model_dir.iterdir())))

    def seed_champion(self, period="p3"):
        adapter = self.tmp_path / "old-adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}")
        run_id = registry.log_candidate(
            "flow-exp", self.metrics_for(period), Decision(True, "x"), adapter
        )
        version = registry.register_version("flow-model", run_id, period)
        registry.tag_version("flow-model", version, hf_repo="me/champions", hf_revision="rev-old")
        registry.set_champion("flow-model", version)

    def metrics_for(self, period):
        scores = {"val": [0.9] * 4, "test": [0.8] * 4}
        return {
            "period": period,
            "periods": PERIODS,
            "best_val_f1": 0.9,
            "train_seconds": 1.0,
            "champion": None,
            "candidate": scores,
        }


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def champion_version():
    return registry.get_champion("flow-model")["version"]


def test_a_drifted_period_with_a_good_candidate_becomes_the_new_champion(world):
    world.seed_champion("p3")

    result = flow_module.retrain_flow("p4", workdir=str(world.tmp_path / "work"))

    assert result["status"] == "promoted"
    assert result["drifted"] and result["drift_score"] > 0.9
    # the job started from the old champion, on the new period only
    assert world.jobs == [
        {
            "commit": "abc123",
            "code_repo": "https://example.com/repo.git",
            "periods": ["p4"],
            "champion": {"repo": "me/champions", "revision": "rev-old"},
        }
    ]
    assert world.uploads == [("me/champions", "champion after p4")]
    assert world.deploys == [("me/champions", "rev-new")]
    assert world.exported == [("me/champions", "rev-new")]
    assert world.space_updates == [("me/demo", "rev-new", ["model.onnx"])]
    champion = registry.get_champion("flow-model")
    assert champion["version"] == "2"
    assert (champion["tags"]["period"], champion["tags"]["hf_revision"]) == ("p4", "rev-new")


def test_no_drift_stops_before_any_training(world, tmp_path, monkeypatch):
    world = World(tmp_path / "calm", monkeypatch, p4_is_new_language=False)
    world.seed_champion("p3")

    result = flow_module.retrain_flow("p4", workdir=str(tmp_path / "work"))

    assert result["status"] == "no drift"
    assert result["drift_score"] < 0.6
    assert world.jobs == [] and world.uploads == [] and world.deploys == []
    assert world.space_updates == []
    assert champion_version() == "1"


def test_a_candidate_that_forgets_is_refused_and_nothing_changes(world):
    world.seed_champion("p3")
    world.scores = FORGETS

    result = flow_module.retrain_flow("p4", workdir=str(world.tmp_path / "work"))

    assert result["status"] == "refused"
    assert "forgets p1" in result["reason"]
    assert world.uploads == [] and world.deploys == [] and world.space_updates == []
    assert champion_version() == "1"
    runs = mlflow.search_runs(experiment_names=["flow-exp"])
    assert (runs["tags.decision"] == "refuse").sum() == 1  # the refusal is on record


def test_the_new_period_must_follow_the_one_the_champion_learned(world):
    world.seed_champion("p3")

    with pytest.raises(ValueError, match="next period"):
        flow_module.retrain_flow("p2", workdir=str(world.tmp_path / "work"))


def test_an_update_needs_a_champion(world):
    with pytest.raises(RuntimeError, match="no champion"):
        flow_module.retrain_flow("p4", workdir=str(world.tmp_path / "work"))


def test_a_failed_switch_says_what_to_do(world):
    world.seed_champion("p3")
    world.deploy_error = OSError("docker is not running")

    with pytest.raises(RuntimeError, match="--deploy-only"):
        flow_module.retrain_flow("p4", workdir=str(world.tmp_path / "work"))

    assert champion_version() == "2"  # the registry already has the new champion


def test_deploy_only_switches_the_api_to_the_registry_champion(world):
    world.seed_champion("p3")

    result = flow_module.deploy_champion_flow()

    assert result == {"status": "deployed", "version": "1"}
    assert world.deploys == [("me/champions", "rev-old")]
    assert world.space_updates == [("me/demo", "rev-old", ["model.onnx"])]


def test_bootstrap_trains_the_first_periods_and_registers_version_one(world):
    result = flow_module.bootstrap_flow("p3", workdir=str(world.tmp_path / "work"))

    assert result["status"] == "promoted"
    assert world.jobs[0]["periods"] == ["p1", "p2", "p3"] and world.jobs[0]["champion"] is None
    champion = registry.get_champion("flow-model")
    assert (champion["version"], champion["tags"]["period"]) == ("1", "p3")
    assert world.deploys == [("me/champions", "rev-new")]
    assert world.space_updates == [("me/demo", "rev-new", ["model.onnx"])]


def test_bootstrap_refuses_to_replace_an_existing_champion(world):
    world.seed_champion("p3")

    with pytest.raises(RuntimeError, match="already a champion"):
        flow_module.bootstrap_flow("p3", workdir=str(world.tmp_path / "work"))


def test_a_failed_demo_update_says_what_to_do_and_keeps_the_new_champion(world):
    world.seed_champion("p3")
    world.space_error = OSError("the Hub is unreachable")

    with pytest.raises(RuntimeError, match="--deploy-only"):
        flow_module.retrain_flow("p4", workdir=str(world.tmp_path / "work"))

    assert world.deploys == [("me/champions", "rev-new")]  # the API was switched already
    assert champion_version() == "2"
