"""Load the sentiment model and turn texts into label probabilities."""

from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.clean import LABELS, normalize_text


@dataclass(frozen=True)
class TextResult:
    probabilities: dict[str, float]
    words: int  # words in the cleaned text
    pieces: int  # word pieces the tokenizer made from it, without [CLS] and [SEP]


class SentimentModel:
    def __init__(self, model: torch.nn.Module, tokenizer, info: dict, max_length: int):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.info = info
        self.max_length = max_length

    @classmethod
    def load(cls, config: dict) -> "SentimentModel":
        tokenizer = AutoTokenizer.from_pretrained(
            config["base_model"], revision=config["base_revision"]
        )
        base = AutoModelForSequenceClassification.from_pretrained(
            config["base_model"], revision=config["base_revision"], num_labels=len(LABELS)
        )
        adapter = PeftModel.from_pretrained(
            base, config["adapter_repo"], revision=config["adapter_revision"]
        )
        # Adding the LoRA weights into the base weights gives a plain BERT classifier, so
        # a prediction costs the same as without LoRA. Training time is the only place
        # where the separate adapter matters.
        model = adapter.merge_and_unload()
        info = {key: config[key] for key in ("base_model", "adapter_repo", "adapter_revision")}
        return cls(model, tokenizer, info, config["max_length"])

    def predict(self, texts: list[str]) -> list[dict[str, float]]:
        """For each text, the probability of every label."""
        return [result.probabilities for result in self.predict_detailed(texts)]

    def predict_detailed(self, texts: list[str]) -> list[TextResult]:
        """Like predict, plus the size of each text in words and in word pieces.

        Slang that the vocabulary does not know is split into more pieces, so pieces per word
        goes up when the language drifts. Counting here reuses the tokenization that the model
        needs anyway.
        """
        # The model was trained on normalized text, so it must be given the same.
        cleaned = [normalize_text(text) for text in texts]
        encoded = self.tokenizer(
            cleaned,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            probabilities = self.model(**encoded).logits.softmax(dim=-1)
        # The attention mask counts real tokens. Two of them are [CLS] and [SEP].
        pieces = (encoded["attention_mask"].sum(dim=1) - 2).tolist()
        return [
            TextResult(dict(zip(LABELS, row, strict=True)), len(text.split()), count)
            for row, text, count in zip(probabilities.tolist(), cleaned, pieces, strict=True)
        ]
