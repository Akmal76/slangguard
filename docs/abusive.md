# The abusive language flag

The sentiment model answers "is this text negative, neutral or positive?". It does not answer
"is this rude?". Rude words can be friendly teasing, and a polite text can be angry. So the API
has a second, separate model, and `POST /predict` returns both answers for each text:
`label` (sentiment) and `abusive` with `abusive_score` (a probability from 0 to 1).

The model card, with usage and limits, is in [abusive_model_card.md](abusive_model_card.md). The
model is [amc76/slangguard-abusive](https://huggingface.co/amc76/slangguard-abusive).

## How it works

TF-IDF over groups of 2 to 5 letters inside words, then logistic regression (scikit-learn). Letter
groups catch spellings like `k0ntol` or `kntl` that a whole-word model has never seen. The file is
1.3 MB, it trains in seconds on a CPU and the API answers with no extra delay. Why not IndoBERT
again: it would need a GPU on Kaggle and add 400 MB to the image for a task that is mostly about
words. Every weight of this model can be read, which also helps to explain a wrong answer.

Code: `src/data/abusive.py` (data), `src/training/abusive.py` (training, settings in
`configs/abusive.yaml`), `src/serving/abusive.py` (loading and scoring).

## Data

The multi-label Indonesian Twitter set of Ibrohim and Budi (2019), column `Abusive`, CC BY-NC-SA
4.0. Raw text is only in the private DVC remote (stage `abusive` in `dvc.yaml`). Cleaning dropped
the source's own placeholders and escape codes, because they leaked the label: the word `USER` is
in 58% of the tweets and 47% of those are abusive (38% overall). It also dropped 146 repeated
tweets. 12,849 tweets are left, split 70/15/15 with seed 42.

## What went wrong first, and the fix

The first model reached **F1 0.894 on its test set** and flagged "kontol" and "memek" with score
1.00, which is what was asked for. Then a hand check found "enak banget" at 0.91. Measured on the
sentiment data of this project, it flagged 8.5% (P1 reviews) to 13.8% (P4 tweets) of all texts,
and up to 15% of the positive ones, and the rate rose with slang, so the detector had its own drift.

The cause: the training tweets are about hate and abuse, so letter groups of "banget" were linked
to abuse, and the model had never seen plain praise labelled as clean. The test score was good
because the test tweets came from the same source. **A test set from the training source does not
tell how the model does on your own text.**

The fix was to add clean text from our own domain: 1,500 texts from the train split of each
sentiment period (6,000 in all), without any text that has a word of the authors' list of 125
abusive words, so that rude tweets are not taught as clean. The target was set before training:
flag rate on positive P1 reviews below 2%, and abusive F1 not more than 0.02 lower.

| | First model | Final model |
| --- | --- | --- |
| F1 abusive, test of the abusive data | 0.894 | 0.893 |
| Positive P1 texts flagged | about 4% (val) | 0.4% (test) |
| All texts flagged, P1 and P4 | 8.5% and 13.8% (val) | 3.7% and 0.7% (test) |
| Hand check (two different sets of sentences), rude flagged / normal flagged | 12 of 12 / 6 of 15 short praise | 14 of 15 / 0 of 18 |

The first revision on the Hub (`ccf9a6dc`) is marked as not to be used in the model card.

## Limits

- It finds abusive words, not intent. Friendly use is flagged and a polite insult is not.
- Very short abbreviations are weak (`anjg` 0.40, `bgst` 0.52 in the hand check).
- The flag rates on the sentiment test sets are not error rates, because those texts have no
  abusive label. Some are really rude. The share with a word of the authors' list is shown next to
  them in the model card.
- The model file is a pickle, so the API loads it only from a fixed revision of our own repo.
- The demo in the browser does not show this flag yet. It would need the TF-IDF written again in
  JavaScript and checked against Python, as was done for the text cleaning.
- The model is CC BY-NC-SA 4.0 (non-commercial) because its data is. The sentiment model keeps
  CC BY-SA 4.0.
