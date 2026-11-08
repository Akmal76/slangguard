# Baseline results

Naive fine-tune and full retrain over P1 to P4, trained on a Kaggle T4 with IndoBERT base
(`configs/train.yaml`). Both runs are logged in MLflow on DagsHub as `naive-seed42` and
`full-seed42`. Row t is the model after step t, and column i is the macro-F1 on the test
set of period i. The bold cells are the period the model was just trained on.

**Status: one seed (42).** Differences of about 0.01 are noise until the 3-seed run is done.

## Naive fine-tune (train only on the new period)

| After | P1 | P2 | P3 | P4 |
| --- | --- | --- | --- | --- |
| step 1 (P1) | **0.929** | 0.543 | 0.522 | 0.470 |
| step 2 (P2) | 0.850 | **0.666** | 0.648 | 0.607 |
| step 3 (P3) | 0.783 | 0.642 | **0.635** | 0.592 |
| step 4 (P4) | 0.786 | 0.673 | 0.648 | **0.643** |

## Full retrain (fresh model on all periods so far)

| After | P1 | P2 | P3 | P4 |
| --- | --- | --- | --- | --- |
| step 1 (P1) | **0.929** | 0.540 | 0.522 | 0.472 |
| step 2 (P1+P2) | 0.922 | **0.688** | 0.635 | 0.473 |
| step 3 (P1..P3) | 0.926 | 0.671 | **0.641** | 0.583 |
| step 4 (P1..P4) | 0.920 | 0.660 | 0.671 | **0.654** |

## Summary

| | Naive | Full retrain |
| --- | --- | --- |
| AvgF1 | 0.687 | 0.726 |
| Forgetting | 0.045 | 0.003 |
| P1 after step 4 | 0.786 | 0.920 |
| Training time per step | 190, 29, 30, 31 s | 194, 239, 281, 326 s |
| Training time, total | 4.7 min | 17.3 min |

## What the numbers show

- **Naive forgets P1.** P1 falls from 0.929 to 0.783 after P3 and stays at 0.786 after P4,
  a loss of about 14 points. This meets the milestone 2 check in the roadmap.
- **Full retrain does not forget.** P1 stays between 0.920 and 0.929 at every step.
- **Drift is visible before any retraining.** The model trained only on P1 scores 0.54,
  0.52 and 0.47 on P2, P3 and P4, and the score falls as the slang gets heavier.
- **The whole AvgF1 gap is P1.** After step 4 the two methods differ by 0.134 on P1 and
  by 0.013, 0.023 and 0.012 on P2 to P4, which is within what one seed can show.
  Naive is just as good as full retrain on the three tweet periods.
- **Cost grows only for full retrain.** Its step time rises from 194 s to 326 s because
  the training set grows, while a naive step stays near 30 s after the first one.
  The last full retrain step costs about 10 times a naive step.

## Limits to keep in mind

1. **The P1 drop is not only slang.** P1 is formal reviews with 11% neutral labels, and
   P2 to P4 are tweets with 50% neutral. Part of what naive forgets is a change of
   domain and label mix. The IDSA-only periods show almost no forgetting among
   themselves (P2 stays at 0.67 after P4), even though P4 has the heaviest slang.
2. **Forgetting is an average over three old periods, and two of them hardly move.**
   It is 0.144 for P1 and about 0 for P2 and P3, so the mean (0.045) hides the P1 loss.
   Report P1 after the last step next to it.
3. **Run-to-run noise.** Step 1 is the same job in both methods (same seed, same data),
   yet P2 and P4 differ by 0.002 between the two runs, from GPU nondeterminism.
4. **The IDSA ceiling is low.** Tweets reach about 0.66 to 0.69 macro-F1 even when the
   model is trained on them, with about 2,100 training tweets per period.
5. **No tuning.** Hyperparameters were fixed before training and shared by both methods.

## Open

- Run seeds 43 and 44, then report mean and standard deviation.
- Revisit the targets in `design.md` section 8 with these numbers.
