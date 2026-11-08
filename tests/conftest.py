import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "enak", "parah", "biasa", "banget"]


@pytest.fixture
def tokenizer(tmp_path):
    """A tiny tokenizer built from a local word list, so tests never download anything."""
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n")
    return BertTokenizerFast(vocab_file=str(vocab_file))


@pytest.fixture
def make_model():
    """Returns a function that builds a fresh, tiny, random BERT classifier."""

    def build(config: dict | None = None) -> BertForSequenceClassification:
        return BertForSequenceClassification(
            BertConfig(
                vocab_size=len(VOCAB),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
                num_labels=3,
            )
        )

    torch.manual_seed(0)
    return build
