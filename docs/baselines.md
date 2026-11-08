# Baseline results

Naive fine-tune and full retrain over P1 to P4, trained on a Kaggle T4 with IndoBERT base
(`configs/train.yaml`), with 3 seeds (42, 43, 44). All runs are in MLflow on DagsHub as
`naive-seed<N>` and `full-seed<N>`. Numbers are mean ± standard deviation over the 3 seeds
(sample standard deviation). F1 is macro-F1 on the test set of each period.

## Summary

| | Naive fine-tune | Full retrain |
| --- | --- | --- |
| AvgF1 | 0.677 ± 0.010 | 0.724 ± 0.003 |
| Forgetting | 0.057 ± 0.012 | 0.001 ± 0.006 |
| P1 F1 after the last step | 0.759 ± 0.039 | 0.925 ± 0.006 |
| P2 F1 after the last step | 0.651 ± 0.020 | 0.661 ± 0.002 |
| P3 F1 after the last step | 0.659 ± 0.013 | 0.668 ± 0.005 |
| P4 F1 after the last step | 0.638 ± 0.006 | 0.643 ± 0.010 |
| Training time, total | 4.9 ± 0.2 min | 17.6 ± 0.2 min |

Per seed (42, 43, 44):

| | Naive | Full retrain |
| --- | --- | --- |
| AvgF1 | 0.687, 0.674, 0.669 | 0.726, 0.721, 0.726 |
| Forgetting | 0.045, 0.055, 0.069 | 0.003, 0.006, -0.006 |
| P1 after the last step | 0.786, 0.777, 0.715 | 0.920, 0.923, 0.932 |

## Mean result matrices

Row t is the model after step t, and column i is the test set of period i. Cells above the
diagonal are periods the model has not seen yet.

Naive fine-tune (trains only on the new period):

| After | P1 | P2 | P3 | P4 |
| --- | --- | --- | --- | --- |
| step 1 (P1) | **0.927** | 0.530 | 0.509 | 0.472 |
| step 2 (P2) | 0.846 | **0.664** | 0.647 | 0.601 |
| step 3 (P3) | 0.800 | 0.642 | **0.634** | 0.609 |
| step 4 (P4) | 0.759 | 0.651 | 0.659 | **0.638** |

Full retrain (a fresh model on all periods so far):

| After | P1 | P2 | P3 | P4 |
| --- | --- | --- | --- | --- |
| step 1 (P1) | **0.927** | 0.530 | 0.509 | 0.472 |
| step 2 (P1+P2) | 0.927 | **0.665** | 0.642 | 0.535 |
| step 3 (P1..P3) | 0.929 | 0.663 | **0.652** | 0.594 |
| step 4 (P1..P4) | 0.925 | 0.661 | 0.668 | **0.643** |

## What the numbers show

- **Naive forgets P1, in every seed.** P1 falls from 0.927 to 0.759 (-16.8 points), and the
  final P1 score is below the full retrain score by 0.13 to 0.22 in all three seeds. This
  meets the milestone 2 check in the roadmap.
- **Full retrain does not forget.** P1 stays between 0.925 and 0.929 at every step.
- **Drift is visible before any retraining.** A model trained only on P1 scores 0.53, 0.51
  and 0.47 on P2, P3 and P4, and the score falls as the slang gets heavier.
- **The AvgF1 gap is 4.7 points (0.039, 0.046, 0.057 per seed) and it comes from P1.** On
  P2 to P4 the final gap between the two methods is 0.010, 0.008 and 0.005 on average, which
  is inside the seed noise. Naive is as good as full retrain on the three tweet periods.
- **Naive forgetting is uneven.** P1 after the last step ranges from 0.715 to 0.786 across
  seeds (standard deviation 0.039), so single-seed claims about forgetting are weak.
- **Cost grows only for full retrain.** A full step takes 200 s at step 1 and about 330 s at
  step 4, because the training set grows. A naive step stays near 31 s after the first one,
  about 10 times cheaper than the last full step.

## Limits to keep in mind

1. **The P1 drop is not only slang.** P1 is formal reviews with 11% neutral labels, and P2
   to P4 are tweets with 50% neutral. Part of what naive forgets is a change of domain and
   label mix. The IDSA-only periods show almost no forgetting among themselves, even though
   P4 has the heaviest slang.
2. **Forgetting is an average over three old periods, and two of them hardly move.** It is
   about 0.17 for P1 and about 0 for P2 and P3, so the mean hides the P1 loss. Report P1
   after the last step next to it.
3. **Run-to-run noise on the GPU.** Step 1 is the same job in both methods (same seed, same
   data), yet some test scores differ by up to 0.002 between the two runs.
4. **Only 3 seeds.** The standard deviation of 3 values is itself rough.
5. **The IDSA ceiling is low.** Tweets reach about 0.66 to 0.67 macro-F1 even when the model
   is trained on them, with about 2,100 training tweets per period.
6. **No tuning.** Hyperparameters were fixed before training and shared by both methods.

## Targets in `design.md`

The first estimates in section 8 hold up against these numbers, so they stay as written:

| Target for LoRA + replay | Threshold from the baselines | Naive today |
| --- | --- | --- |
| AvgF1 at most 3 points below full retrain | at least 0.694 | 0.677 (fails) |
| Forgetting less than half of naive | below 0.029 | 0.057 |
| Training time per update less than half of full retrain | below about 165 s per update | 31 s |

The AvgF1 margin between naive and the threshold is only 1.7 points, about 1.7 standard
deviations of naive, so LoRA + replay needs to be compared over several seeds too.
