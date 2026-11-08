"""Load the sentiment model and turn texts into label probabilities."""

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.clean import LABELS, normalize_text


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
        # The model was trained on normalized text, so it must be given the same.
        encoded = self.tokenizer(
            [normalize_text(text) for text in texts],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.inference_mode():
            probabilities = self.model(**encoded).logits.softmax(dim=-1)
        return [dict(zip(LABELS, row, strict=True)) for row in probabilities.tolist()]
