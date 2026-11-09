"""Turn the champion into a model that runs in a web browser.

The merged model is exported to ONNX and its weights are stored as 8-bit integers, which makes
the file about four times smaller. The folder has the layout that Transformers.js reads, so a
static web page can load it without a server.

    uv run python -m src.pipelines.browser_model --out web/model
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.data.clean import LABELS, normalize_text
from src.serving.config import load_serving_config
from src.serving.model import SentimentModel

INPUTS = ["input_ids", "attention_mask", "token_type_ids"]


def export_onnx(model: torch.nn.Module, tokenizer, path: Path) -> None:
    """Write the model as an ONNX file. Batch size and text length can vary."""
    model.eval()
    sample = tokenizer(["enak banget", "parah"], padding=True, return_tensors="pt")
    args = tuple(sample[name] for name in INPUTS)
    axes = {name: {0: "batch", 1: "tokens"} for name in INPUTS} | {"logits": {0: "batch"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        args,
        str(path),
        input_names=INPUTS,
        output_names=["logits"],
        dynamic_axes=axes,
        dynamo=False,
    )


def quantize(source: Path, destination: Path) -> None:
    """Store the weights as int8. The numbers a text goes through stay in floating point."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(source), str(destination), weight_type=QuantType.QInt8)


def write_browser_model(model, tokenizer, out_dir: Path, info: dict) -> dict[str, float]:
    """Write the folder that the web page loads. Returns the file sizes in MB."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        full = Path(temp) / "model.onnx"
        export_onnx(model, tokenizer, full)
        quantize(full, out_dir / "onnx" / "model_quantized.onnx")
        full_mb = full.stat().st_size / 1e6
    # The base model config lists five labels. This project has three, in this order.
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = {label: i for i, label in enumerate(LABELS)}
    model.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "model-info.json").write_text(json.dumps(info, indent=2))
    return {
        "full_mb": full_mb,
        "int8_mb": (out_dir / "onnx" / "model_quantized.onnx").stat().st_size / 1e6,
    }


def predict_onnx(path: Path, tokenizer, texts: list[str]) -> np.ndarray:
    """Label probabilities from an exported model, to check it against the original."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    encoded = tokenizer([normalize_text(t) for t in texts], padding=True, return_tensors="np")
    wanted = {i.name for i in session.get_inputs()}
    logits = session.run(["logits"], {k: v for k, v in encoded.items() if k in wanted})[0]
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def export_champion(config: dict, out_dir: Path) -> dict[str, float]:
    """Load the model named in a serving config, merge it and write the browser folder."""
    sentiment = SentimentModel.load(config)
    info = {
        key: config[key]
        for key in ("base_model", "base_revision", "adapter_repo", "adapter_revision")
    }
    info["weights"] = "int8"
    return write_browser_model(sentiment.model, sentiment.tokenizer, out_dir, info)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/serving.yaml"))
    args = parser.parse_args()

    # ADAPTER_REPO and ADAPTER_REVISION in the environment choose the model, like for the API.
    sizes = export_champion(load_serving_config(args.config, os.environ), args.out)
    print(f"written to {args.out}: {sizes['full_mb']:.0f} MB as float,", end=" ")
    print(f"{sizes['int8_mb']:.0f} MB as int8")


if __name__ == "__main__":
    main()
