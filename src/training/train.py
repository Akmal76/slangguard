"""Fine-tune a sequence classifier for one baseline step. Runs on CPU and on CUDA."""

import copy
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from src.data.clean import LABELS
from src.data.tsv import Row
from src.evaluation.metrics import macro_f1
from src.training.data import LABEL_TO_ID


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA


def new_model(config: dict) -> torch.nn.Module:
    # The classification head is always new and random. Some checkpoints (like the tiny test
    # model) ship a head with another size, which would otherwise raise an error.
    return AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=len(LABELS), ignore_mismatched_sizes=True
    )


def make_loader(rows: list[Row], tokenizer, config: dict, shuffle: bool) -> DataLoader:
    def collate(batch: list[Row]):
        texts, labels = zip(*batch, strict=True)
        # Padding to the longest text in the batch, not to max_length, saves compute.
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=config["max_length"],
            return_tensors="pt",
        )
        return encoded, torch.tensor([LABEL_TO_ID[label] for label in labels])

    return DataLoader(rows, batch_size=config["batch_size"], shuffle=shuffle, collate_fn=collate)


def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> list[str]:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for encoded, _ in loader:
            logits = model(**encoded.to(device)).logits
            predictions += [LABELS[i] for i in logits.argmax(dim=-1).tolist()]
    return predictions


def score(model, rows: list[Row], tokenizer, config: dict, device: torch.device) -> float:
    """Macro-F1 of the model on the given rows."""
    predictions = predict(model, make_loader(rows, tokenizer, config, shuffle=False), device)
    return macro_f1([label for _, label in rows], predictions)


def train_step(model, tokenizer, train: list[Row], val: list[Row], config: dict, device) -> float:
    """Train for the configured epochs, then restore the epoch with the best validation F1.

    Returns that best validation F1. Test sets are never touched here.
    """
    model.to(device)
    loader = make_loader(train, tokenizer, config, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    total_steps = len(loader) * config["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(config["warmup_ratio"] * total_steps), total_steps
    )
    # Half precision is only used on GPU, where it is faster. The scaler stops tiny
    # gradients from rounding to zero in half precision.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_weights = -1.0, None
    for _ in range(config["epochs"]):
        model.train()
        for encoded, labels in loader:
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**encoded.to(device), labels=labels.to(device)).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # clip real gradient sizes, not scaled ones
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        val_f1 = score(model, val, tokenizer, config, device)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return best_f1
