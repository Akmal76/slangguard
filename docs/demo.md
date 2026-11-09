# The public demo

The demo is a web page where anyone can type an Indonesian sentence and get its sentiment:
<https://huggingface.co/spaces/amc76/slangguard-demo>. **The model runs in the visitor's browser.**
There is no server, so it costs nothing and nothing the visitor types leaves their computer.

## Why a browser model

The first plan was a Gradio app on a Hugging Face Space. Creating it failed with `402 Payment
Required`: Hugging Face now says that Gradio and Docker Spaces on the free CPU need a paid plan, and
only static Spaces are free. This project has no budget, so the model moved into the page instead.

## How it works

1. The champion (IndoBERT plus the LoRA adapter) is merged into one model and exported to ONNX
   (`src/pipelines/browser_model.py`).
2. The weights are stored as 8-bit integers (dynamic quantization). The file goes from 498 MB to
   125 MB.
3. The page (`web/`) loads it with [Transformers.js](https://github.com/huggingface/transformers.js)
   (ONNX Runtime Web) from a pinned CDN version, cleans the text exactly like `normalize_text` in
   Python, and shows the three probabilities.
4. When the pipeline promotes a new champion it exports and uploads the model to the Space too, so the
   demo follows the API. The page shows which revision it serves.

## What was checked

All numbers below are from the champion after P4 (registry version 2).

**The export is faithful.** The float ONNX model gives the same label as the PyTorch model on every
text of the 8 validation and test sets (agreement 1.0000, probability difference below 1e-5).

**int8 costs almost nothing in accuracy.** Over the 8 sets the mean macro-F1 is 0.7229 with int8 and
0.7218 without, a change of +0.0011 (worst single set -0.0050, best +0.0097). That is below the
noise between seeds (about 0.01). int8 changes the label of 0.2% to 2.2% of texts, depending on the
set.

**The page matches Python.**

| Check | Result |
| --- | --- |
| Cleaning text in JavaScript against Python (21 tricky strings) | identical (a test that runs when Node.js is installed, as on CI) |
| Tokenizer, 1,600 validation texts | 0 differ |
| Model in Node, 1,600 texts, against Python int8 | 13 labels differ (0.8%) |
| Model in a real browser, 200 texts, against Python int8 | 3 labels differ (1.5%), no page errors |
| The six examples on the live page, against the float model | same labels, probabilities within about 2 points |

The labels that differ are texts the model is unsure about. For the 13 in the Node check, the gap
between the two most likely labels was 0.018 at the median, against 0.72 for a typical text. Different
ONNX Runtime versions round 8-bit numbers a little differently.

Macro-F1 on those 1,600 validation texts: 0.7341 for PyTorch, 0.7317 for Python int8 and 0.7360 for
the JavaScript model.

## Limits and things to know

- **First visit.** The page downloads about 125 MB, which took about 45 seconds on this connection.
  The browser may keep the file, so later visits can be faster: on localhost a second load took
  under a second, but this was not measured on the public page, and the caching was not checked.
  A prediction takes about 0.1 s.
- **The JavaScript library is pinned to 3.8.1 on purpose.** Version 4.3.2, the latest, failed in a real
  browser when loaded straight from the CDN: its default file broke the tokenizer, and its web build
  needs an import map. A test checks that the page stays on a 3.x version.
- **The page depends on a CDN** (jsDelivr) for the library and the WebAssembly runtime.
- **int8** changes the label of about 1% of texts, as measured above.
- The slang drift in the experiments is simulated, and the model reaches about 0.65 to 0.69 macro-F1
  on tweets. It should not be used for decisions about people.
- The examples on the page are written by hand. A test checks that none of them is a sentence from the
  dataset, because that text may not be shared.

## Running it yourself

```bash
uv run python -m src.pipelines.browser_model --out web/model    # export the champion (about 25 s)
cd web && python3 -m http.server 8765                           # then open http://localhost:8765
```

`web/model/` is ignored by git because it is about 125 MB. There is also a Gradio app for trying the
model on this computer: `PYTHONPATH=. uv run python demo/app.py`.
