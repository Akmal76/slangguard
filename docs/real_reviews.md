# Real shopper reviews

Until now the model was tested on simulated periods. This page tests it on text written by real
shoppers: **5,257 Tokopedia product reviews** from PRDECT-ID (Sutoyo et al., *Data in Brief*, 2022,
CC BY 4.0). The model never trained or tuned on them. The text stays in the private DVC remote
(stage `real_reviews`), and only the numbers are published.

## Rules, written before the first run

The rules and targets are in `configs/real_reviews.yaml`, and the evaluation ran once.

- PRDECT-ID has two labels, positive and negative. The model has three. A **neutral prediction counts
  as a miss**, so the score punishes a model that hides in the middle.
- Score: macro-F1 over positive and negative, with a 95% interval from 1,000 bootstrap resamples.
- The label follows the star rating (405 of the 462 reviews with 3 stars are "negative", although many
  read as mixed). So the score is also given **without 3-star reviews**.
- Targets for the model that people use (the 8-bit champion in the browser demo): **0.85** on all
  reviews and **0.90** without 3 stars.
- Cleaning: same `normalize_text` as in training. 143 of 5,400 reviews dropped: 135 duplicate texts and
  8 texts that also appear in a train or val split of the periods.

## Results

| Model | All reviews | Without 3 stars | Predicted neutral |
| --- | --- | --- | --- |
| **Browser demo (champion v2, 8-bit)** | **0.924** (0.918 to 0.930) | **0.933** (0.927 to 0.939) | 9.8% |
| Same champion, full precision | 0.923 | 0.932 | 10.0% |
| API model (LoRA + replay, seed 42) | 0.921 | 0.930 | 10.8% |

Both targets pass. The three models are within about 0.003 of each other, which is below the noise
between seeds, so none of them is better on this set. 8-bit costs nothing here.

The abusive flag marks 1.9% of all reviews and 0.7% of the positive ones. These reviews have no
abusive label, so these are flag rates and not error rates.

## What this does not show

- **One platform, one kind of text.** Tokopedia reviews are close to the SmSA reviews the model trained
  on (P1). The score says the model works on product reviews. It says nothing yet about YouTube,
  Instagram or Shopee comments, where slang is heavier. That is the next thing to test.
- **Labels follow the stars**, so a review with a rude word and 5 stars counts as positive.
- **Near-duplicates may remain.** Short reviews like "bagus" or "sesuai pesanan" appear many times.
  Only exact copies and training overlaps were removed, so the score can be a little high.
- The 10% of reviews the model calls neutral are counted as misses. Many of them are the 3-star
  reviews, so part of that cost comes from the label and not from the model.

Code: `src/data/real_reviews.py`, `src/evaluation/real_reviews.py`. Numbers: `results/real_reviews.json`.
