# Results

One JSON file per training run, named `<method>[-r<replay %>]-seed<N>.json`. They hold
metrics only, no dataset text, so they are safe to keep in git. The same numbers are in
MLflow on DagsHub.

Fields: `matrix[t][i]` is the macro-F1 on the test set of period i after step t (see
`src/evaluation/metrics.py`), `step_seconds` is the training time of each step, and
`avg_f1`, `forgetting` and `total_train_seconds` are the summaries.

The six Week 2 runs (`naive` and `full`, seeds 42 to 44) were added by hand from the Kaggle
output. Their `step_seconds` for seeds 43 and 44 come from the run log and are rounded to
whole seconds, and `trainable_params` is null because it was not logged back then.
