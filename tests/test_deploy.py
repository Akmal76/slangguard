import pytest

from src.pipelines import deploy as deploy_module
from src.pipelines.deploy import deploy, wait_for_revision, write_champion_env
from src.serving.config import load_serving_config

CONFIG_TEXT = "adapter_repo: user/original\nadapter_revision: aaa\nmax_texts: 32\n"


def health(revision):
    return {"status": "ok", "model": {"adapter_revision": revision}}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(deploy_module.time, "sleep", lambda seconds: None)


def test_the_config_is_used_when_nothing_is_overridden(tmp_path):
    path = tmp_path / "serving.yaml"
    path.write_text(CONFIG_TEXT)

    config = load_serving_config(path, {})

    assert (config["adapter_repo"], config["adapter_revision"]) == ("user/original", "aaa")


def test_environment_variables_replace_the_model_but_not_other_settings(tmp_path):
    path = tmp_path / "serving.yaml"
    path.write_text(CONFIG_TEXT)

    config = load_serving_config(path, {"ADAPTER_REPO": "user/new", "ADAPTER_REVISION": "bbb"})

    assert (config["adapter_repo"], config["adapter_revision"]) == ("user/new", "bbb")
    assert config["max_texts"] == 32


def test_empty_environment_values_are_ignored(tmp_path):
    path = tmp_path / "serving.yaml"
    path.write_text(CONFIG_TEXT)

    config = load_serving_config(path, {"ADAPTER_REPO": "", "ADAPTER_REVISION": ""})

    assert config["adapter_revision"] == "aaa"


def test_champion_env_is_written_and_overwritten(tmp_path):
    path = tmp_path / "champion.env"

    write_champion_env(path, "user/one", "111")
    write_champion_env(path, "user/two", "222")

    assert path.read_text() == "ADAPTER_REPO=user/two\nADAPTER_REVISION=222\n"


def test_wait_returns_once_the_new_revision_is_served():
    answers = iter([None, health("old"), health("old"), health("new")])

    wait_for_revision("http://x", "new", fetch=lambda url: next(answers))


def test_wait_gives_up_if_the_old_model_keeps_answering():
    with pytest.raises(TimeoutError, match="last seen: old"):
        wait_for_revision("http://x", "new", timeout_seconds=0, fetch=lambda url: health("old"))


def test_deploy_writes_the_setting_recreates_the_container_and_waits(tmp_path):
    commands = []
    answers = iter([health("old"), health("222")])

    deploy(
        "user/two",
        "222",
        env_file=tmp_path / "champion.env",
        run=lambda command, check: commands.append(command),
        fetch=lambda url: next(answers),
    )

    assert commands == [["docker", "compose", "up", "-d", "--build", "--force-recreate", "api"]]
    assert (
        tmp_path / "champion.env"
    ).read_text() == "ADAPTER_REPO=user/two\nADAPTER_REVISION=222\n"
