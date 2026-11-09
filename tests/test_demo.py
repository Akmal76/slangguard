import importlib.util
from pathlib import Path

import gradio as gr
import pytest

from src.data.clean import normalize_text
from src.data.tsv import read_tsv
from src.serving.model import SentimentModel

INFO = {"base_model": "tiny", "adapter_repo": "me/champions", "adapter_revision": "abcdef123456"}
PERIODS = Path("data/periods")


def load_demo_module():
    spec = importlib.util.spec_from_file_location("demo_app", Path("demo/app.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def demo_app():
    return load_demo_module()


@pytest.fixture
def model(make_model, tokenizer):
    return SentimentModel(make_model(), tokenizer, INFO, max_length=8)


def test_classify_returns_a_probability_for_each_label(demo_app, model):
    result = demo_app.classify(model, "enak banget", max_chars=100)

    assert list(result) == ["negative", "neutral", "positive"]
    assert sum(result.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("text", ["", "   ", "!!!"])
def test_text_without_words_gets_a_readable_error(demo_app, model, text):
    with pytest.raises(gr.Error, match="at least one word"):
        demo_app.classify(model, text, max_chars=100)


def test_text_that_is_too_long_gets_a_readable_error(demo_app, model):
    with pytest.raises(gr.Error, match="at most 10 characters"):
        demo_app.classify(model, "enak " * 5, max_chars=10)


def test_the_demo_builds_and_names_the_model_it_serves(demo_app, model):
    demo = demo_app.build_demo(model, max_chars=100)

    assert isinstance(demo, gr.Blocks)
    assert "abcdef12" in str(demo.get_config_file())  # the revision is shown to the user


def test_the_examples_are_the_demos_own_sentences_and_all_have_words(demo_app):
    assert len(demo_app.EXAMPLES) >= 4
    assert all(normalize_text(example) for example in demo_app.EXAMPLES)


@pytest.mark.skipif(not PERIODS.exists(), reason="needs the period data (dvc pull)")
def test_no_example_is_a_sentence_from_the_dataset(demo_app):
    """The dataset text may not be shared, so a copy of it must not end up in a public demo."""
    dataset_texts = {
        normalize_text(text) for path in PERIODS.glob("p?/*.tsv") for text, _ in read_tsv(path)
    }

    copied = [e for e in demo_app.EXAMPLES if normalize_text(e) in dataset_texts]

    assert copied == []
