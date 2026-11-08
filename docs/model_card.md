---
language: id
license: cc-by-sa-4.0
library_name: peft
base_model: indobenchmark/indobert-base-p1
pipeline_tag: text-classification
tags:
  - lora
  - peft
  - sentiment-analysis
  - indonesian
  - continual-learning
---

# SlangGuard IndoBERT LoRA + replay

A LoRA adapter for [`indobenchmark/indobert-base-p1`](https://huggingface.co/indobenchmark/indobert-base-p1)
that classifies Indonesian text as **negative, neutral or positive**. It comes from
[SlangGuard](https://github.com/Akmal76/slangguard), a portfolio project on keeping a
sentiment model up to date when slang changes, without forgetting older language.

The adapter was trained **one period at a time** (P1, then P2, P3 and P4), each time on the
new period plus a 5% replay sample of the older ones. Only 297,219 weights (0.24% of the model)
were trained: the LoRA matrices on the attention query and value layers, and the
classification head. The file is about 1.2 MB.

## How to use

The model expects text that was normalized like the training text: lowercase, no punctuation,
no standalone `amp`. Labels are `0` negative, `1` neutral and `2` positive.

```python
import re

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

base = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=3)
model = PeftModel.from_pretrained(model, "amc76/slangguard-indobert-lora-replay").eval()


def normalize(text):
    text = re.sub(r"[^\w\s]", " ", text.lower().replace("'", ""))
    return " ".join(re.sub(r"\bamp\b", " ", text).split())


texts = ["makanannya enak banget, pelayanannya ramah", "parah sih, lama banget nunggunya"]
batch = tokenizer(
    [normalize(t) for t in texts],
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)
with torch.inference_mode():
    labels = model(**batch).logits.argmax(dim=-1).tolist()
print([["negative", "neutral", "positive"][i] for i in labels])
```

The `num_labels=3` line prints a warning about `id2label`, because the base model config lists
five labels. It is harmless: the classification head is part of this adapter.

## Training data

| Period | Source | Text |
| --- | --- | --- |
| P1 | SmSA from [IndoNLU](https://github.com/IndoNLP/indonlu) (Wilie et al., 2020) | formal reviews and comments |
| P2 | Indonesian Sentiment Twitter Dataset (IDSA), [ridife/dataset-idsa](https://github.com/ridife/dataset-idsa) | tweets |
| P3 | IDSA, a different part | tweets, with words replaced by slang forms at a moderate rate |
| P4 | IDSA, a different part | tweets, with a high rate of slang forms plus 100 words written in reverse |

The slang forms come from the Colloquial Indonesian Lexicon
([nasalsabila/kamus-alay](https://github.com/nasalsabila/kamus-alay), Salsabila et al., 2018),
used in the formal-to-slang direction. The authors of IDSA and of the lexicon agreed to the use
of their data in this project and to the publication of models trained on it. **The dataset
text is not included here** and is not redistributed. Cleaning removed punctuation and `amp`,
and dropped tweets that contain common Malay words (a word-list filter, not language detection).

**P3 and P4 are simulated.** They are real tweets rewritten with a lexicon, not language that
was collected later. The results below show forgetting under simulated drift, not under real
language change.

## Results

Macro-F1 on the held-out test set of each period, for this adapter (seed 42), after training
on all four periods:

| P1 | P2 | P3 | P4 |
| --- | --- | --- | --- |
| 0.905 | 0.687 | 0.689 | 0.645 |

Over 3 seeds, compared with the two reference methods (mean ± standard deviation):

| Method | AvgF1 over the 4 periods | P1 after the last update | Training time of the last update |
| --- | --- | --- | --- |
| Naive fine-tune, new period only | 0.677 ± 0.010 | 0.759 ± 0.039 | 32 s |
| Full retrain on all periods | 0.724 ± 0.003 | 0.925 ± 0.006 | 328 s |
| **LoRA + 5% replay (this method)** | **0.725 ± 0.006** | **0.900 ± 0.006** | **29 s** |

This adapter is the seed 42 run, chosen before any result was seen. Its AvgF1 is 0.731, a bit
above the average of the method. Full details and limits are in the
[results](https://github.com/Akmal76/slangguard/blob/main/docs/results.md).

## Limits

- **Indonesian only**, and tuned for short informal text such as tweets and reviews. Other
  domains were not tested.
- **Tweets reach only about 0.65 to 0.69 macro-F1**, even when the model was trained on them.
  The labels are noisy and there are few training tweets per period (about 2,100).
- **The old period is the formal one.** Part of what simple fine-tuning forgets is a change of
  domain and label mix (neutral is 11% of P1 and about 50% of the tweet periods), not only slang.
- **Not tuned.** All settings were fixed before training. Other settings could change the numbers.
- **Not for decisions about people.** Sentiment labels on social media text are subjective, and
  the model can reflect biases in the data. Do not use it where a wrong label has a real cost.

## Citation

The base model and SmSA come from the IndoNLU benchmark: Wilie et al., "IndoNLU: Benchmark and
Resources for Evaluating Indonesian Natural Language Understanding", 2020. Please also credit
the IDSA and Colloquial Indonesian Lexicon authors linked above.
