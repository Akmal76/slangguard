---
title: SlangGuard
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: cc-by-sa-4.0
short_description: Indonesian sentiment that keeps up with slang
---

# SlangGuard demo

A web page that labels Indonesian text as negative, neutral or positive. The model runs in your
browser (ONNX Runtime Web through Transformers.js), so there is no server and nothing you type is
sent anywhere.

- Project, design and results: <https://github.com/Akmal76/slangguard>
- The model behind it: an [IndoBERT](https://huggingface.co/indobenchmark/indobert-base-p1) LoRA
  adapter promoted by the SlangGuard pipeline. The revision in use is shown on the page.

The files of this Space are copied from the project by `src/pipelines/space.py`. Do not edit them here.
