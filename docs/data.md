# Data Survey

Candidate datasets for the drift periods, checked on 8 Nov 2026.
Every candidate needs 3 sentiment classes (negative, neutral, positive) and a license
that allows the use described here.

## 1. Candidates

| Dataset | Source | Text style | Labels | Rows | License | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| SmSA | [IndoNLU](https://github.com/IndoNLP/indonlu) | Formal reviews and comments | 3 classes | 12,760 | Open, see note 1 | **Use** |
| Indonesian Sentiment Twitter Dataset (IDSA) | [ridife/dataset-idsa](https://github.com/ridife/dataset-idsa) | Informal tweets, up to 2019 | 3 classes (-1, 0, 1) | 10,806 | None stated, see note 2 | **Use, with limits** |
| Colloquial Indonesian Lexicon | [nasalsabila/kamus-alay](https://github.com/nasalsabila/kamus-alay) | Slang to formal word pairs | Not labeled | 15,006 pairs, 4,331 slang words | None stated, see note 2 | **Use, with limits** |
| NusaX-Senti (Indonesian) | [IndoNLP/nusax](https://github.com/IndoNLP/nusax) | Formal reviews | 3 classes | 1,000 | CC BY-SA 4.0 | Drop, see note 3 |
| Indonesian Google Play reviews | [Hugging Face](https://huggingface.co/datasets/RimuruChan67/indonesian-googleplay-sentiment) | Informal app reviews | 2 classes, from star ratings | 5,336 | CC BY 4.0 | Drop: no neutral class |
| IndoLEM sentiment | [indolem/indolem](https://github.com/indolem/indolem) | Tweets and hotel reviews | 2 classes | 5,048 | CC BY-SA 4.0 | Drop: no neutral class |
| `carant-ai/indonesian_sentiment_dataset` | [Hugging Face](https://huggingface.co/datasets/carant-ai/indonesian_sentiment_dataset) | Mixed, merged from 12 sources | 3 classes | 1,030,393 | None stated | Drop: unclear license, already contains SmSA |
| `sepidmnorozy/Indonesian_sentiment` | [Hugging Face](https://huggingface.co/datasets/sepidmnorozy/Indonesian_sentiment) | Unknown | 2 classes | 11,324 | None stated | Drop: no neutral class, unknown source |

## 2. Label distribution

| Dataset | Negative | Neutral | Positive |
| --- | --- | --- | --- |
| SmSA (train + valid + test) | 4,034 (32%) | 1,367 (11%) | 7,359 (58%) |
| IDSA | 2,887 (27%) | 5,327 (49%) | 2,592 (24%) |

The neutral share changes from 11% to 49% between the two datasets. So the drift is not
only in vocabulary but also in the label mix (label shift). Macro-F1 weighs every class
equally, which keeps the metric fair, but the analysis must report both effects
separately.

## 3. Notes

1. **SmSA license.** Sources disagree. The IndoNLU README and the Hugging Face card say
   MIT, the repository `LICENSE` file says Apache-2.0, and the SEACrowd catalog lists
   CC BY-SA 4.0. This project follows the strictest one, CC BY-SA 4.0: give credit, and
   share derived data under the same license.
2. **No license stated (IDSA, lexicon).** Without a license, the authors keep all rights
   by default. Tweet text is also covered by the X Terms of Service. So this project:
   - downloads these files from the original source inside the pipeline,
   - never publishes the raw or transformed text in this repository or a public remote,
   - keeps the DVC remote private,
   - cites the original authors.

   **Permission (8 Nov 2026).** Both authors were emailed and agreed to the use of their
   data in this project and to publishing the models trained on it. This does not cover
   redistributing the text, so the rules above stay: the data remains on the private remote,
   and only trained models (such as the LoRA adapter on Hugging Face Hub) are published.
   The emails are kept by the project owner and are not part of the repository.
3. **NusaX overlaps with SmSA.** The NusaX card states it was sampled from SmSA. A check
   found that 287 of the 1,000 Indonesian rows match SmSA text exactly after lowercasing,
   9 of them inside the SmSA validation and test sets. Using it as a new period would
   leak data and add no style change.

## 4. Proposed periods

| Period | Data | Style |
| --- | --- | --- |
| P1 | SmSA | Formal |
| P2 | IDSA, part A | Real informal tweets |
| P3 | IDSA, part B, rewritten with the lexicon at a moderate rate | Heavier slang |
| P4 | IDSA, part C, rewritten at a high rate plus made-up new words | Simulated future slang |

- IDSA is split into three parts with no shared tweets, so no text appears in two periods.
- The lexicon maps slang to formal words. The rewrite uses it in the other direction:
  a formal word is replaced with one of its slang forms.
- Each period gets its own train, validation, and test split with a fixed seed.
- P3 and P4 are simulated. The README must say so.

## 5. Cleaning rules

Applied in `src/data/clean.py`, before the period split. Found in
`notebooks/01_data_analysis.ipynb`.

| Rule | Why | Effect |
| --- | --- | --- |
| Lowercase, collapse spaces | One text format | n/a |
| Strip punctuation: apostrophes are joined, other marks become spaces | SmSA has punctuation and IDSA has none, so a model could tell P1 from later periods by punctuation alone | `teman-teman` becomes `teman teman`; `__laugh__` is kept |
| Remove the standalone token `amp` | Leftover of `&amp;`, not a word | 148 texts changed in the first count |
| Drop IDSA tweets with a Malay marker word | The project is about Indonesian | 973 tweets before deduplication |
| Drop empty texts and texts with conflicting labels | Unusable or contradictory | See `deduplicate` |

Known limits:

- **Malay filter is a word list, not language detection.** Markers are in
  `configs/data.yaml` (`malay_markers`). `macam` is not a marker because 254 SmSA texts
  use it as Indonesian. About 1% of P2 (31 tweets) still has `macam`, so some Malay
  may remain, and a few Indonesian tweets that use a marker word are lost.
- **Dropped Malay tweets were not skewed.** The IDSA label mix hardly moved
  (negative 26.2% before, 25.6% after).
- **Punctuation can carry sentiment** ("!" and "?" in P1). Removing it costs a little
  signal, but it is the same cost in every period.
- **Hyphens are split**, so reduplication (`benar-benar`) becomes two words.
