# Model ladder benchmark: 500 patients (tag ladder500_imp_reconcile)

Same dataset, patient-level split and metrics as the linear baseline.
Within-patient normalization: none (primary baseline).

| Model | In-domain near-optimal | In-domain regret | Shifted near-optimal | Shifted regret |
|---|---:|---:|---:|---:|
| Linear | 0.640 | 0.0180 | 0.640 | 0.0217 |
| Linear no sensors | 0.493 | 0.0366 | 0.493 | 0.0366 |
| LambdaMART features | 0.680 | 0.0248 | 0.600 | 0.0317 |
| Tiny CNN raw signal | 0.107 | 0.1561 | 0.107 | 0.1467 |
| LambdaMART CNN embed | 0.667 | 0.0240 | 0.507 | 0.0361 |
| LambdaMART hybrid | 0.693 | 0.0236 | 0.587 | 0.0248 |
| Linear +CNN embed | 0.800 | 0.0109 | 0.680 | 0.0205 |
| Blend linear+embed | 0.800 | 0.0109 | 0.680 | 0.0205 |

## Near-optimal fraction at 5% tolerance

| Model | In-domain near-optimal | Shifted near-optimal |
|---|---:|---:|
| Linear | 0.880 | 0.867 |
| Linear no sensors | 0.680 | 0.680 |
| LambdaMART features | 0.813 | 0.747 |
| Tiny CNN raw signal | 0.160 | 0.213 |
| LambdaMART CNN embed | 0.827 | 0.733 |
| LambdaMART hybrid | 0.827 | 0.827 |
| Linear +CNN embed | 0.947 | 0.893 |
| Blend linear+embed | 0.947 | 0.893 |

## GBT vs linear, paired bootstrap (95% CI, test patients)

- source near-optimal: +0.040 [-0.093, +0.160]
- source regret: +0.0067 [-0.0025, +0.0171]
- shifted near-optimal: -0.040 [-0.173, +0.093]
- shifted regret: +0.0100 [+0.0006, +0.0208]

## Negative control

- GBT with within-patient shuffled labels: near-optimal 0.080, Spearman -0.149.

## GBT learning curve (test-set regret vs train patients)

| n train patients | near-optimal | regret |
|---:|---:|---:|
| 10 | 0.373 | 0.0594 |
| 20 | 0.453 | 0.0521 |
| 40 | 0.493 | 0.0549 |
| 80 | 0.480 | 0.0511 |
| 150 | 0.547 | 0.0357 |
| 350 | 0.693 | 0.0238 |

## Top GBT features by gain

| Feature | gain fraction |
|---|---:|
| ix_set_av_x_vv_x_ctx_hr | 0.436 |
| ix_set_vv_sq_x_ctx_intrinsic_av_est | 0.162 |
| set_vv | 0.067 |
| ix_set_av_x_ctx_hr | 0.052 |
| ctx_hr | 0.028 |
| imp_stroke | 0.025 |
| ix_set_vv_x_imp_stroke | 0.022 |
| ix_set_av_sq_x_ctx_hr | 0.018 |
| ix_set_vv_x_ctx_qrs_est | 0.018 |
| ix_set_vv_sq_x_ctx_qrs_est | 0.018 |

In-silico demonstration only; not clinical validation.
