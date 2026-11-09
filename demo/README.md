---
title: SlangGuard
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.13"
app_file: app.py
pinned: false
license: cc-by-sa-4.0
short_description: Indonesian sentiment that keeps up with slang
---

# SlangGuard demo

Type an Indonesian sentence and get its sentiment: negative, neutral or positive.

The model is [IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1) with a LoRA adapter.
The adapter is the *champion* of the SlangGuard retraining pipeline: when new text drifts away
from what the model has seen, the pipeline trains a new adapter on a GPU, tests it against the
current one, and promotes it only if it learns the new language without forgetting the old.
This Space follows the champion. Its `ADAPTER_REPO` and `ADAPTER_REVISION` variables are set by the
pipeline, and the revision in use is shown under the result.

- Project, design and results: <https://github.com/Akmal76/slangguard>
- Released model with a full model card: <https://huggingface.co/amc76/slangguard-indobert-lora-replay>

**Limits.** The slang drift in the experiments is simulated, the model reaches about 0.65 to 0.69
macro-F1 on tweets, and it should not be used for decisions about people. This app does not store
what you type, but Hugging Face may keep access logs.

The code in this Space is copied from the project by `src/pipelines/space.py`. Do not edit it here.
