import importlib

import pytest

SUBPACKAGES = ["data", "training", "evaluation", "drift", "pipelines", "serving"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(f"src.{name}")
