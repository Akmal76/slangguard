# Results: LoRA + replay against the baselines

Seven methods over P1 to P4, each trained with 3 seeds (42, 43, 44) on a Kaggle T4 with
IndoBERT base. Numbers are mean ± sample standard deviation over seeds, and F1 is macro-F1 on
the test set of each period. The baselines are described in [baselines.md](baselines.md).
The per-run files are in `results/`, the runs are in MLflow on DagsHub, and the tables and
charts can be rebuilt with `notebooks/03_results.ipynb`.

| Method | Trains | AvgF1 | Forgetting | P1 after P4 | Total time | Last step |
| --- | --- | --- | --- | --- | --- | --- |
| Naive fine-tune | all weights, new period | 0.677 ± 0.010 | 0.057 ± 0.012 | 0.759 ± 0.039 | 4.9 min | 32 s |
| LoRA | 0.24% of weights, new period | 0.703 ± 0.003 | 0.025 ± 0.009 | 0.836 ± 0.016 | 3.1 min | 17 s |
| Replay 5% | all weights, new + buffer | 0.706 ± 0.004 | 0.023 ± 0.014 | 0.881 ± 0.010 | 5.7 min | 50 s |
| **LoRA + replay 1%** | 0.24% of weights, new + buffer | 0.720 ± 0.006 | 0.008 ± 0.007 | 0.874 ± 0.016 | 3.2 min | 21 s |
| **LoRA + replay 5%** | 0.24% of weights, new + buffer | 0.725 ± 0.006 | -0.002 ± 0.010 | 0.900 ± 0.006 | 3.7 min | 29 s |
| **LoRA + replay 10%** | 0.24% of weights, new + buffer | 0.726 ± 0.008 | -0.007 ± 0.007 | 0.913 ± 0.001 | 4.0 min | 38 s |
| Full retrain | all weights, all periods, new model | 0.724 ± 0.003 | 0.001 ± 0.006 | 0.925 ± 0.006 | 17.6 min | 328 s |

"Last step" is the training time of the update that adds P4. "Total time" includes the first
step, which trains the initial model on P1 in every method. The LoRA runs train 297,219 of
about 124.4 million weights.

## What the numbers show

1. **LoRA + replay matches full retrain on AvgF1.** With a 5% buffer it scores 0.725 against
   0.724, and per seed the difference is +0.005, +0.004 and -0.007, so there is no clear gap
   either way. Against naive fine-tuning the gain is +0.048 (+0.044, +0.050, +0.050 per seed).
2. **It does so at a fraction of the cost.** The update that adds P4 takes 29 s against 328 s
   for full retrain, about 11 times less, and the cost per update does not grow with the number
   of periods the way full retrain does (243, 285 and 328 s at steps 2 to 4 against about 27 to
   29 s). Over the whole sequence it is 3.7 minutes against 17.6.
3. **Both parts help, and neither alone reaches full retrain.** Against naive, LoRA alone adds +0.026 AvgF1
   and replay alone adds +0.029, while together they add +0.048. For P1 after the last step the
   gains are +0.077 (LoRA), +0.122 (replay) and +0.141 (both). So the ablations answer the
   interview question: replay does most of the work for the old period, and LoRA adds a smaller
   gain on top of it while also making each update cheaper.
4. **LoRA alone is also cheaper than naive.** It runs a step in 15 to 17 s against 30 to 32 s,
   and the first step in 136 s against 198 s.
5. **The buffer size has a clear effect only between 1% and 5%.** On P1 after the last step
   the order is 0.874 (1%), 0.900 (5%) and 0.913 (10%). On AvgF1 the three are within one
   standard deviation of each other (0.720, 0.725, 0.726). A 5% buffer is a good default, and
   going to 10% costs 9 more seconds in the last step for a small gain on P1.
6. **The remaining gap to full retrain is on P1.** LoRA + replay 5% ends at 0.900 against
   0.925, and the gap is 0.014, 0.021 and 0.039 in the three seeds. On P2 to P4 it is as good as
   full retrain (0.682, 0.675 and 0.644 against 0.661, 0.668 and 0.643).

## Targets from `design.md` section 8

| Target | Limit from the baselines | LoRA + replay 5% | Result |
| --- | --- | --- | --- |
| AvgF1 at most 3 points below full retrain | at least 0.694 | 0.725 | pass |
| Forgetting less than half of naive | below 0.028 | -0.002 | pass |
| Training time per update below half of full retrain | below 164 s | 29 s | pass |

The other targets (API latency and the automated pipeline) belong to later milestones.

These three targets are loose. LoRA alone and replay alone also pass them (AvgF1 0.703 and
0.706, forgetting 0.025 and 0.023), so they do not tell the proposed method from its
ablations. P1 after the last step does: 0.836, 0.881 and 0.900 for LoRA, replay and the
combination. Report it next to the three targets.

## Limits to keep in mind

1. **Three seeds.** The standard deviation of three values is rough, and several differences
   above (5% against 10%, LoRA against replay alone) are inside it. Treat only the large gaps
   as findings: naive against everything else, and full retrain's cost.
2. **The P1 drop is not only slang.** P1 is formal reviews with 11% neutral labels, and P2 to
   P4 are tweets with 50% neutral. Part of what naive fine-tuning forgets is a change of domain
   and label mix, and replay of P1 rows helps with that as much as with slang.
3. **Replay needs stored old data.** The 5% buffer holds about 100 tweets of each old period.
   A real system must keep and protect them. The LoRA adapter alone does not need them.
4. **Not tuned, and not equal.** All settings were fixed before the runs. LoRA uses a learning
   rate of 2e-4 and full fine-tuning 2e-5, each a common default for its kind, and neither was
   searched. Tuning could change the numbers, so the ranking is for these settings.
5. **Time is training time.** It covers the training loop and the validation after each epoch,
   not loading the model or scoring the four test sets. Runs were sequential on one GPU.
6. **Periods P3 and P4 are simulated** from P2-style tweets with a slang lexicon, so these
   results show forgetting under simulated drift, not under real language change.
7. **The IDSA ceiling is low.** Tweets stay near 0.65 to 0.68 macro-F1 for every method.

## The model to publish

Chosen before any result was known: LoRA + replay with a 5% buffer, seed 42. Its AvgF1 is 0.731
(the three seeds gave 0.731, 0.725 and 0.719), so it is slightly above the mean of its method.
The adapter file is about 1.2 MB and holds only the LoRA matrices and the classification head.
It is published at [amc76/slangguard-indobert-lora-replay](https://huggingface.co/amc76/slangguard-indobert-lora-replay)
under CC BY-SA 4.0, with the model card in [model_card.md](model_card.md).
