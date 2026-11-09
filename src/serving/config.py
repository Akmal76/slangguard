"""Read the serving settings. Used by the API and by the demo, so both pick the same model."""

from collections.abc import Mapping
from pathlib import Path

import yaml

# Environment variables that replace the model in configs/serving.yaml. The pipeline sets them
# (through champion.env for the API, and as a variable of the Space for the demo) when it
# promotes a new model.
ENV_OVERRIDES = {"ADAPTER_REPO": "adapter_repo", "ADAPTER_REVISION": "adapter_revision"}


def load_serving_config(path: Path, env: Mapping[str, str]) -> dict:
    config = yaml.safe_load(path.read_text())
    for name, key in ENV_OVERRIDES.items():
        if env.get(name):  # an empty value counts as not set
            config[key] = env[name]
    return config
