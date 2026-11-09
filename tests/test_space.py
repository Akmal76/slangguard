import ast
from pathlib import Path

import pytest

from src.pipelines import space as space_module
from src.pipelines.space import (
    DEMO_FILES,
    SOURCES,
    assemble,
    point_at_champion,
    publish,
    wait_until_running,
)

ROOT = Path(".")


class FakeApi:
    def __init__(self, stages=()):
        self.calls = []
        self.stages = list(stages)

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))

    def add_space_variable(self, repo_id, key, value):
        self.calls.append(("variable", key, value))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload", kwargs["repo_id"], kwargs["repo_type"]))

    def restart_space(self, repo_id):
        self.calls.append(("restart", repo_id))

    def get_space_runtime(self, repo_id):
        self.calls.append(("stage", repo_id))
        return type("Runtime", (), {"stage": self.stages.pop(0)})()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(space_module.time, "sleep", lambda seconds: None)


def test_assemble_copies_exactly_the_listed_files(tmp_path):
    names = assemble(ROOT, tmp_path)

    on_disk = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert on_disk == sorted([*DEMO_FILES, *SOURCES])
    assert sorted(names) == on_disk


def test_the_space_gets_no_data_and_no_notebooks(tmp_path):
    assemble(ROOT, tmp_path)

    files = [p.name for p in tmp_path.rglob("*") if p.is_file()]
    assert not any(name.endswith((".tsv", ".csv", ".ipynb", ".dvc")) for name in files)


def test_the_copied_code_is_the_projects_code(tmp_path):
    assemble(ROOT, tmp_path)

    for name in ("src/serving/model.py", "src/data/clean.py", "configs/serving.yaml"):
        assert (tmp_path / name).read_text() == (ROOT / name).read_text()


def imported_project_modules(folder: Path) -> set[str]:
    modules = set()
    for path in folder.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src"):
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules |= {a.name for a in node.names if a.name.startswith("src")}
    return modules


def test_every_project_module_the_space_imports_is_in_the_space(tmp_path):
    """A forgotten file would only show up as a failed build on Hugging Face."""
    assemble(ROOT, tmp_path)

    for module in imported_project_modules(tmp_path):
        relative = Path(*module.split("."))
        assert (tmp_path / relative).with_suffix(".py").exists() or (
            tmp_path / relative / "__init__.py"
        ).exists(), module


def test_the_space_requirements_pin_every_package_the_model_needs():
    text = (ROOT / "demo/requirements.txt").read_text()

    for package in ("torch==", "transformers==", "peft==", "pyyaml=="):
        assert package in text


def test_publish_creates_a_gradio_space_and_sets_the_model_before_uploading(tmp_path):
    api = FakeApi()

    publish(api, "me/demo", tmp_path, "me/champions", "rev1")

    kinds = [call[0] for call in api.calls]
    assert kinds == ["create_repo", "variable", "variable", "upload"]
    assert api.calls[0][2] == {"repo_type": "space", "space_sdk": "gradio", "exist_ok": True}
    assert [call[1:] for call in api.calls[1:3]] == [
        ("ADAPTER_REPO", "me/champions"),
        ("ADAPTER_REVISION", "rev1"),
    ]
    assert api.calls[3] == ("upload", "me/demo", "space")


def test_point_at_champion_sets_both_variables_and_restarts():
    api = FakeApi()

    point_at_champion(api, "me/demo", "me/champions", "rev2")

    assert api.calls == [
        ("variable", "ADAPTER_REPO", "me/champions"),
        ("variable", "ADAPTER_REVISION", "rev2"),
        ("restart", "me/demo"),
    ]


def test_wait_returns_when_the_space_runs():
    api = FakeApi(["BUILDING", "APP_STARTING", "RUNNING"])

    wait_until_running(api, "me/demo")

    assert len(api.calls) == 3


def test_wait_raises_when_the_build_fails():
    with pytest.raises(RuntimeError, match="BUILD_ERROR"):
        wait_until_running(FakeApi(["BUILDING", "BUILD_ERROR"]), "me/demo")


def test_wait_gives_up_after_the_timeout():
    with pytest.raises(TimeoutError):
        wait_until_running(FakeApi(["BUILDING"] * 3), "me/demo", timeout_seconds=0)
