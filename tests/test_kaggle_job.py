import ast
import subprocess

import pytest

from src.pipelines import kaggle_job

CFG = {
    "owner": "me",
    "dataset": "periods-data",
    "kernel": "update-job",
    "docker_image": "image@sha256:abc",
    "poll_seconds": 0,
    "timeout_minutes": 10,
}
SPEC = {
    "commit": "abc1234",
    "code_repo": "https://example.com/repo.git",
    "periods": ["p4"],
    "champion": None,
}


class FakeKaggle:
    """Stands in for the Kaggle tool. Answers are taken from `replies`, one per call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, *args, check=True):
        self.calls.append(args)
        returncode, stdout = self.replies.pop(0) if self.replies else (0, "")
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(kaggle_job.time, "sleep", lambda seconds: None)


def install(monkeypatch, replies):
    fake = FakeKaggle(replies)
    monkeypatch.setattr(kaggle_job, "kaggle", fake)
    return fake


def status(name):
    return (0, f'me/update-job has status "KernelWorkerStatus.{name}"')


def test_rendered_script_is_python_that_holds_the_settings_and_the_job_code():
    script = kaggle_job.render_script(SPEC)

    tree = ast.parse(script)  # raises if the script is not valid Python
    spec = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and node.target.id == "SPEC"
    )
    assert spec == SPEC
    assert spec["champion"] is None
    assert "def main" in script


def test_prepare_dataset_flattens_the_period_files(tmp_path):
    periods = tmp_path / "periods"
    for period in ("p1", "p2"):
        (periods / period).mkdir(parents=True)
        for split in ("train", "val", "test"):
            (periods / period / f"{split}.tsv").write_text(f"{period} {split}")

    kaggle_job.prepare_dataset(periods, tmp_path / "out", "me", "periods-data")

    names = sorted(path.name for path in (tmp_path / "out").iterdir())
    assert names == [
        "dataset-metadata.json",
        *[f"{p}_{s}.tsv" for p in ("p1", "p2") for s in ("test", "train", "val")],
    ]
    assert (tmp_path / "out" / "p2_val.tsv").read_text() == "p2 val"
    assert '"id": "me/periods-data"' in (tmp_path / "out" / "dataset-metadata.json").read_text()


def test_kernel_metadata_asks_for_a_gpu_and_the_dataset():
    metadata = kaggle_job.kernel_metadata(CFG)

    assert metadata["enable_gpu"] and metadata["kernel_type"] == "script"
    assert metadata["id"] == "me/update-job"
    assert metadata["dataset_sources"] == ["me/periods-data"]
    assert metadata["docker_image"] == "image@sha256:abc"


def test_a_new_dataset_is_created_and_an_existing_one_gets_a_version(monkeypatch, tmp_path):
    fake = install(monkeypatch, [(1, ""), (0, ""), (0, "ready")])
    kaggle_job.upload_dataset(tmp_path, "me", "periods-data", "msg")
    assert [call[:2] for call in fake.calls] == [
        ("datasets", "status"),
        ("datasets", "create"),
        ("datasets", "status"),
    ]

    fake = install(monkeypatch, [(0, "ready"), (0, ""), (0, "ready")])
    kaggle_job.upload_dataset(tmp_path, "me", "periods-data", "msg")
    assert fake.calls[1][:2] == ("datasets", "version")


def test_upload_waits_until_the_dataset_is_ready(monkeypatch, tmp_path):
    fake = install(
        monkeypatch, [(0, "ready"), (0, ""), (0, "pending"), (0, "pending"), (0, "ready")]
    )

    kaggle_job.upload_dataset(tmp_path, "me", "periods-data", "msg")

    assert len(fake.calls) == 5


def test_push_kernel_writes_the_script_and_pushes_it(monkeypatch, tmp_path):
    fake = install(monkeypatch, [])

    kaggle_job.push_kernel(CFG, SPEC, tmp_path / "kernel")

    assert fake.calls == [("kernels", "push", "-p", str(tmp_path / "kernel"))]
    assert "abc1234" in (tmp_path / "kernel" / "script.py").read_text()
    assert (tmp_path / "kernel" / "kernel-metadata.json").exists()


def test_wait_returns_when_the_job_completes(monkeypatch):
    fake = install(monkeypatch, [status("RUNNING"), status("RUNNING"), status("COMPLETE")])

    kaggle_job.wait_for_kernel(CFG)

    assert len(fake.calls) == 3


def test_wait_raises_when_the_job_fails(monkeypatch):
    install(monkeypatch, [status("RUNNING"), status("ERROR")])

    with pytest.raises(RuntimeError, match="ERROR"):
        kaggle_job.wait_for_kernel(CFG)


def test_wait_gives_up_after_the_timeout(monkeypatch):
    install(monkeypatch, [status("RUNNING")] * 5)

    with pytest.raises(TimeoutError):
        kaggle_job.wait_for_kernel({**CFG, "timeout_minutes": 0})


def test_fetch_results_downloads_the_kernel_output(monkeypatch, tmp_path):
    fake = install(monkeypatch, [])

    folder = kaggle_job.fetch_results(CFG, tmp_path / "output")

    assert fake.calls == [("kernels", "output", "me/update-job", "-p", str(tmp_path / "output"))]
    assert folder == tmp_path / "output" / "result"
