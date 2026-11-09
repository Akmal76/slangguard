"""A demo to run on this computer, with the same model code as the API.

The public demo is the web page in web/ (Gradio Spaces are no longer free). This one is for
trying the model locally, from the project folder:

    PYTHONPATH=. uv run python demo/app.py
"""

import json
import os
from pathlib import Path

import gradio as gr

from src.data.clean import normalize_text
from src.serving.config import load_serving_config
from src.serving.model import SentimentModel

# The same sentences as the web page, written by hand. They are not taken from any dataset,
# because that text may not be shared.
EXAMPLES = json.loads(Path("web/examples.json").read_text())

DESCRIPTION = """
Sentiment of Indonesian text: negative, neutral or positive. The model is IndoBERT with a small
LoRA adapter that the [SlangGuard](https://github.com/Akmal76/slangguard) pipeline retrains when
the language of new text drifts, so it keeps up with slang without forgetting older language.

**Limits.** The slang drift in the experiments is simulated (real tweets rewritten with a slang
lexicon). The model is meant for short informal text, reaches about 0.65 to 0.69 macro-F1 on
tweets, and should not be used for decisions about people. This app does not store what you type.
"""


def classify(model: SentimentModel, text: str, max_chars: int) -> dict[str, float]:
    """The probability of each label for one text. Raises a message the user can read."""
    if len(text) > max_chars:
        raise gr.Error(f"Please use at most {max_chars} characters.")
    if not normalize_text(text):
        raise gr.Error("Please write at least one word.")
    return model.predict([text])[0]


def build_demo(model: SentimentModel, max_chars: int) -> gr.Blocks:
    info = model.info
    with gr.Blocks(title="SlangGuard") as demo:
        gr.Markdown("# SlangGuard: sentiment for Indonesian text")
        gr.Markdown(DESCRIPTION)
        text = gr.Textbox(label="Text", lines=3, placeholder="Tulis kalimat di sini...")
        button = gr.Button("Classify", variant="primary")
        result = gr.Label(num_top_classes=3, label="Sentiment")
        gr.Examples(examples=EXAMPLES, inputs=text)
        gr.Markdown(
            f"Model: [`{info['adapter_repo']}`](https://huggingface.co/{info['adapter_repo']}) "
            f"at revision `{info['adapter_revision'][:8]}`"
        )
        for trigger in (button.click, text.submit):
            trigger(
                lambda value: classify(model, value, max_chars), text, result, api_name="predict"
            )
    return demo


if __name__ == "__main__":
    config = load_serving_config(Path("configs/serving.yaml"), os.environ)
    build_demo(SentimentModel.load(config), config["max_chars"]).launch()
