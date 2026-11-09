import json

import numpy as np
import pytest
import torch

from src.data.clean import LABELS
from src.pipelines.browser_model import export_onnx, predict_onnx, quantize, write_browser_model

TEXTS = ["enak banget", "parah", "Biasa, BANGET!"]
INFO = {"adapter_repo": "me/champions", "adapter_revision": "abc123", "weights": "int8"}


@pytest.fixture
def model(make_model):
    model = make_model().eval()
    with torch.no_grad():  # a head that is not at its start value, so the labels differ
        model.classifier.weight.normal_(std=1.0)
    return model


def torch_probabilities(model, tokenizer, texts):
    from src.data.clean import normalize_text

    encoded = tokenizer([normalize_text(t) for t in texts], padding=True, return_tensors="pt")
    with torch.inference_mode():
        return model(**encoded).logits.softmax(dim=-1).numpy()


def test_the_exported_model_gives_the_same_probabilities_as_the_original(
    model, tokenizer, tmp_path
):
    export_onnx(model, tokenizer, tmp_path / "model.onnx")

    exported = predict_onnx(tmp_path / "model.onnx", tokenizer, TEXTS)

    np.testing.assert_allclose(exported, torch_probabilities(model, tokenizer, TEXTS), atol=1e-4)


def test_the_exported_model_accepts_any_batch_size_and_text_length(model, tokenizer, tmp_path):
    export_onnx(model, tokenizer, tmp_path / "model.onnx")

    one = predict_onnx(tmp_path / "model.onnx", tokenizer, ["enak"])
    many = predict_onnx(tmp_path / "model.onnx", tokenizer, ["enak banget parah biasa"] * 5)

    assert one.shape == (1, 3) and many.shape == (5, 3)


def test_the_int8_model_is_smaller_and_still_gives_valid_probabilities(model, tokenizer, tmp_path):
    export_onnx(model, tokenizer, tmp_path / "model.onnx")
    quantize(tmp_path / "model.onnx", tmp_path / "small.onnx")

    probabilities = predict_onnx(tmp_path / "small.onnx", tokenizer, TEXTS)

    assert (tmp_path / "small.onnx").stat().st_size < (tmp_path / "model.onnx").stat().st_size
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_the_browser_folder_has_what_the_page_loads(model, tokenizer, tmp_path):
    sizes = write_browser_model(model, tokenizer, tmp_path / "web-model", INFO)

    folder = tmp_path / "web-model"
    for name in (
        "onnx/model_quantized.onnx",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        assert (folder / name).exists(), name
    assert sizes["int8_mb"] < sizes["full_mb"]
    assert json.loads((folder / "model-info.json").read_text()) == INFO


def test_the_label_names_in_the_config_follow_the_project_order(model, tokenizer, tmp_path):
    write_browser_model(model, tokenizer, tmp_path / "web-model", INFO)

    config = json.loads((tmp_path / "web-model" / "config.json").read_text())

    assert config["id2label"] == {str(i): label for i, label in enumerate(LABELS)}
    assert config["label2id"] == {label: i for i, label in enumerate(LABELS)}
