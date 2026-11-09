from pathlib import Path

import pytest

from src.pipelines import space as space_module
from src.pipelines.space import PAGE_FILES, assemble, publish, update_model, wait_until_running

ROOT = Path(".")


class FakeApi:
    def __init__(self, stages=()):
        self.calls = []
        self.stages = list(stages)

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(
            ("upload", kwargs.get("path_in_repo"), kwargs["repo_id"], kwargs["commit_message"])
        )

    def get_space_runtime(self, repo_id):
        self.calls.append(("stage", repo_id))
        return type("Runtime", (), {"stage": self.stages.pop(0)})()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(space_module.time, "sleep", lambda seconds: None)


@pytest.fixture
def model_dir(tmp_path):
    folder = tmp_path / "exported"
    (folder / "onnx").mkdir(parents=True)
    (folder / "onnx" / "model_quantized.onnx").write_bytes(b"weights")
    (folder / "config.json").write_text("{}")
    return folder


def test_assemble_puts_the_page_and_the_model_in_one_folder(tmp_path, model_dir):
    names = assemble(ROOT, model_dir, tmp_path / "space")

    assert set(PAGE_FILES) <= set(names)
    assert "model/onnx/model_quantized.onnx" in names
    assert "model/config.json" in names
    assert len(names) == len(PAGE_FILES) + 2


def test_the_copied_page_is_the_projects_page(tmp_path, model_dir):
    assemble(ROOT, model_dir, tmp_path / "space")

    for name in PAGE_FILES:
        assert (tmp_path / "space" / name).read_text() == (ROOT / "web" / name).read_text()


def test_the_space_gets_no_dataset_files(tmp_path, model_dir):
    names = assemble(ROOT, model_dir, tmp_path / "space")

    assert not any(name.endswith((".tsv", ".csv", ".ipynb", ".dvc")) for name in names)


def test_publish_creates_a_static_space_and_uploads_everything(tmp_path):
    api = FakeApi()

    publish(api, "me/demo", tmp_path)

    assert api.calls[0] == (
        "create_repo",
        "me/demo",
        {"repo_type": "space", "space_sdk": "static", "exist_ok": True},
    )
    assert api.calls[1][:3] == ("upload", None, "me/demo")


def test_update_model_replaces_only_the_model_folder_and_names_the_revision(tmp_path):
    api = FakeApi()

    update_model(api, "me/demo", tmp_path, "abcdef1234567890")

    assert api.calls == [("upload", "model", "me/demo", "Serve champion revision abcdef12")]


def test_wait_returns_when_the_space_runs():
    api = FakeApi(["BUILDING", "RUNNING"])

    wait_until_running(api, "me/demo")

    assert len(api.calls) == 2


def test_wait_raises_when_the_build_fails():
    with pytest.raises(RuntimeError, match="BUILD_ERROR"):
        wait_until_running(FakeApi(["BUILDING", "BUILD_ERROR"]), "me/demo")


def test_wait_gives_up_after_the_timeout():
    with pytest.raises(TimeoutError):
        wait_until_running(FakeApi(["BUILDING"] * 3), "me/demo", timeout_seconds=0)
