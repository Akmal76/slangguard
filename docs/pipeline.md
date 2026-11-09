# The retraining pipeline

When a new period of text arrives, the pipeline decides whether the language has changed, and if it
has, trains a new model, checks it against the current one, and switches the API only when the new
model passes. The code is in `src/pipelines/`, `src/drift/` and `src/training/update.py`, and the
settings are in `configs/pipeline.yaml`.

```mermaid
flowchart TD
    A[New period of text] --> B{Drift check<br/>Evidently}
    B -- no drift --> Z[Stop, nothing changes]
    B -- drift --> C[Kaggle GPU: train a copy of the champion<br/>on the new period + replay sample]
    C --> D{Promotion rule<br/>validation scores only}
    D -- refused --> L[Log the candidate and the reason]
    D -- passed --> E[Publish adapter on Hugging Face]
    E --> F[Register version, move alias champion]
    F --> G[Restart the API on the new revision]
```

## Status

The code and its tests are done (every step is tested, with fakes only for Kaggle, Hugging Face and
Docker). **The first real run on Kaggle and DagsHub has not been done yet**, so the results of a
real update (bootstrap champion, then P4) are not in this document. Parts that have run against
the real service: the Evidently check on the real period data, the local API switch in Docker,
and Prefect's local server. Parts that have not: the Kaggle dataset and job, the MLflow registry on
DagsHub (it is tested on a local MLflow), and the upload to the pipeline repository on Hugging Face.

## The steps

**1. Drift check** (`src/drift/detect.py`). Evidently trains a small classifier to tell the text of
the newest period the champion learned from the new text. Its ROC AUC is the drift score: 0.5 means
the two look the same, 1 means they are easy to tell apart. It needs no labels. Evidently's own flag
only says the difference is not luck, and with enough texts a tiny difference passes. So a period
counts as drifted only if that flag is set and the score is above a threshold of 0.60.
The threshold comes from controls on the period data, 5 random samples of 1,500 texts each:

| Compared | Mean AUC | Range | Flagged by Evidently |
| --- | --- | --- | --- |
| Same period, train against val (P2, P3, P4) | 0.49 to 0.52 | 0.48 to 0.54 | 0 of 5 each |
| P2 to P3 (light slang rewrite) | 0.663 | 0.65 to 0.69 | 5 of 5 |
| P1 to P2 (reviews to tweets) | 0.922 | 0.91 to 0.93 | 5 of 5 |
| P3 to P4 (heavy slang) | 0.976 | 0.97 to 0.98 | 5 of 5 |

The threshold sits between the highest control (0.543) and the smallest real change (0.648). The
periods are simulated, so on real text the numbers will differ and the threshold needs the same
check again.

**2. Update step** (`src/training/update.py`). It loads the champion adapter and scores it on the
validation and test sets of every period before any training. Then it trains a copy on the new
period plus the replay sample of the older ones (the same sample as in the experiments, rebuilt
from the seed, so it never has to be stored), keeps the best epoch by the validation score of the
new period, and scores the candidate on all validation and test sets. The champion's files are never
changed. On Kaggle this takes about 30 seconds of training for one period.

**3. Promotion rule** (`src/pipelines/promote.py`). The candidate must beat the champion on the
validation set of the new period, and no older period may lose more than 0.02 macro-F1 on its
validation set. Test sets are scored and logged but never decide anything. A refused candidate is
logged with the reason and is not registered.

**4. Registry** (`src/pipelines/registry.py`, MLflow on DagsHub). Every update is a run with its
scores, decision and adapter. Passing candidates become a model version, tagged with the Hugging
Face repo and revision, and the alias `champion` points to the version in use. **Promotion moves the
alias, and a rollback moves it back.**

**5. Switching the API** (`src/pipelines/deploy.py`). The flow writes the new repo and revision to
`champion.env`, recreates the container and waits until `/health` shows the new revision. The API is
down for about 10 seconds. If the switch fails after the alias moved, the flow says so, and
`--deploy-only` retries the switch.

## Running it

One-time setup: copy `.env.example` to `.env` and put the DagsHub token in it, log in to Hugging Face
(`uv run hf auth login`), and have the Kaggle token in `~/.kaggle/`. The code has to be committed and
pushed, because the Kaggle job clones it from GitHub.

```bash
uv run --env-file .env python -m src.pipelines.flow --bootstrap p3     # first champion, P1 to P3
uv run --env-file .env python -m src.pipelines.flow --new-period p4    # an update
uv run --env-file .env python -m src.pipelines.flow --deploy-only      # restart the API
```

Prefect needs no separate server: it starts a local one on the first run (about 15 seconds). To see
the runs in a browser, run `uv run prefect server start` in another terminal.

## The Kaggle job

The period files go to a private Kaggle dataset in the project owner's account, and the job is a
script kernel with a GPU (`src/pipelines/kaggle_body.py`). **The job holds no secret**: the code is
cloned from a public GitHub commit, the champion is downloaded from a public Hugging Face repo, and
the data comes from the private dataset. This avoids the problem of Kaggle dropping secrets on a
push from the command line. The job stops by itself when it finishes or fails. The Kaggle tool can
not cancel a job, so the pipeline gives up waiting after 45 minutes and says so, and a stuck job
has to be stopped on kaggle.com.

## Limits

- **The drift is simulated**, and the threshold is calibrated on that simulation.
- **The candidate's validation score for the new period is a little optimistic**, because the same
  validation set picked its best epoch. The champion has no such advantage. The rule still asks for
  a gain above zero, and the test scores are reported next to it.
- **A 2-point limit on forgetting is a choice**, and with validation sets of a few hundred texts a
  difference of that size can be noise.
- **Only the newest learned period is the drift reference.** Slow drift across many small steps
  would not be seen if each step is below the threshold.
- **One champion at a time**, no shadow deployment or gradual rollout, and the API restarts.
