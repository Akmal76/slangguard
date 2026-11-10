---
language: id
license: cc-by-nc-sa-4.0
library_name: scikit-learn
pipeline_tag: text-classification
tags:
  - abusive-language
  - indonesian
  - tf-idf
  - logistic-regression
---

# SlangGuard abusive language detector

A small model that says whether an Indonesian text contains **abusive language** (kata kasar).
It is the second model of [SlangGuard](https://github.com/Akmal76/slangguard), a portfolio
project. It is separate from the sentiment model, because the two questions differ: "ujaran ini
negatif?" and "ujaran ini kasar?" have different answers for friendly teasing and for polite anger.

It is a scikit-learn pipeline: TF-IDF over groups of 2 to 5 letters inside words
(`char_wb`), then logistic regression. Letter groups let it catch spellings like `k0ntol` and
`kntl` that a whole-word model has never seen. The file is 1.3 MB and runs on a CPU.

## How to use

The text must be normalized like the training text: lowercase, no punctuation, no standalone
`amp`, one space between words. The function is `normalize_text` in
[`src/data/clean.py`](https://github.com/Akmal76/slangguard/blob/main/src/data/clean.py).

```python
import joblib
from huggingface_hub import hf_hub_download

path = hf_hub_download("amc76/slangguard-abusive", "model.joblib", revision="<commit>")
model = joblib.load(path)  # a pickle: load only a revision you trust
score = model.predict_proba(["dasar goblok lu"])[:, 1]  # probability of "abusive"
```

**A pickle can run code when it is loaded.** Pin the revision and do not load files from repos
you do not trust.

## Training data

**Abusive and clean tweets.** [Multi-label hate speech and abusive language detection in Indonesian Twitter](https://github.com/okkyibrohim/id-multi-label-hate-speech-and-abusive-language-detection)
(Ibrohim and Budi, 2019), column `Abusive`, **CC BY-NC-SA 4.0**. After cleaning there are 12,849
tweets (38% abusive), split 70/15/15 with seed 42: 8,994 train, 1,927 validation and 1,928 test.
The text of the data is not included here.

Cleaning removed what the source added and a real user would never type: the placeholders
`USER`, `URL` and `RT`, the escape codes `\n` and `\x..`, and non-ASCII characters. These were
not noise: `USER` is in 58% of the tweets and 47% of them are abusive (38% overall), so a model
could have learned it instead of words. Duplicates and texts with two different labels were dropped.

**Extra clean text.** That data is Twitter text about hate and abuse. A model trained on it alone
learned that intensifiers such as "banget" mean abuse, and flagged plain praise ("enak banget"
scored 0.91; 6 of 15 short praise sentences were flagged). So 6,000 more texts labelled
*not abusive* were added to training: 1,500 from the train split of each of the four sentiment
periods of SlangGuard (reviews and tweets). Texts with a word of the authors' list of 125 abusive
words were left out of this sample, so that a rude tweet is not taught as clean.

## Results

The setting `C` was chosen on a validation set only: the abusive validation tweets plus 1,200
clean texts from the sentiment validation splits (30, from a grid of 0.3 to 100). The test sets
were not used to choose anything.

| Set | F1 (abusive) | Precision | Recall | Macro-F1 |
| --- | --- | --- | --- | --- |
| Abusive data, validation | 0.889 | 0.915 | 0.864 | 0.912 |
| Abusive data, test | 0.893 | 0.911 | 0.877 | 0.915 |

That test set is the same kind of text as the training data. For texts like those SlangGuard
serves, here is how often the detector flags the sentiment **test** sets (they have no abusive
label, and some of the texts really are rude, so this is not an error rate):

| Period | Flagged | Flagged among positive texts | Has a word of the authors' list |
| --- | --- | --- | --- |
| P1 | 3.7% | 0.4% | 5.7% |
| P2 | 2.2% | 1.8% | 3.9% |
| P3 | 1.8% | 2.7% | 3.3% |
| P4 | 0.7% | 0.9% | 2.0% |

A hand check of 33 sentences written for this card: 14 of 15 rude ones were flagged (`anjg`
scored 0.40 and `bgst` 0.52) and none of 18 normal ones. That check is not a measurement.

## Limits

- **It finds abusive words, not intent.** Friendly use among friends is flagged, and a polite
  insult is not. Context is not read.
- **Very short abbreviations are weak** (`anjg`, `bgst`), and some words on the list are
  ambiguous (`babi` in a menu).
- **The data is Twitter from 2018 and 2019, plus the sentiment text of this project.** Newer slang
  and other platforms will do worse.
- **It is not a hate speech detector.** Only the `Abusive` label was used.
- Do not use it alone to moderate or punish people. It is a signal for a human or a filter.

## History

The first revision of this repo (`ccf9a6dc`) was trained on the abusive tweets alone and flagged
8.5% to 13.8% of the sentiment texts, including plain praise. It is replaced by the revision
described above. Do not use `ccf9a6dc`.

## Citation

Ibrohim, M. O. and Budi, I. 2019. Multi-label Hate Speech and Abusive Language Detection in
Indonesian Twitter. In Proceedings of the Third Workshop on Abusive Language Online, pp. 46-57.

The model is licensed **CC BY-NC-SA 4.0**, the same as its training data: attribution,
non-commercial use, and the same license for derived work.
