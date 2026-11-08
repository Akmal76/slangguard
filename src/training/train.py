"""Fine-tune a sequence classifier for one baseline step. Runs on CPU and on CUDA."""

import random

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
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


def add_lora(model: torch.nn.Module, config: dict) -> PeftModel:
    """Freeze the model and add small trainable LoRA matrices to the attention layers.

    PEFT also keeps the classification head trainable for this task type.
    """
    settings = config["lora"]
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=settings["r"],
        lora_alpha=settings["alpha"],
        lora_dropout=settings["dropout"],
        target_modules=settings["target_modules"],
    )
    return get_peft_model(model, lora_config)


def new_lora_model(config: dict) -> PeftModel:
    return add_lora(new_model(config), config)


def trainable_params(model: torch.nn.Module) -> tuple[int, int]:
    """Number of weights that training changes, and the number of all weights."""
    params = list(model.parameters())
    return sum(p.numel() for p in params if p.requires_grad), sum(p.numel() for p in params)


def learning_rate(model: torch.nn.Module, config: dict) -> float:
    # The LoRA update starts at zero and adds only a few weights, so it needs a much larger
    # step than full fine-tuning.
    return (
        config["lora"]["learning_rate"] if isinstance(model, PeftModel) else config["learning_rate"]
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
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate(model, config), weight_decay=config["weight_decay"]
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
            torch.nn.utils.clip_grad_norm_(trainable, config["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        val_f1 = score(model, val, tokenizer, config, device)
        if val_f1 > best_f1:
            best_f1 = val_f1
            # Only the weights that training changes need to be kept.
            best_weights = {
                n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
            }

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in best_weights:
                param.copy_(best_weights[name])
    return best_f1
