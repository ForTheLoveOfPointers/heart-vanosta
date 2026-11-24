# Phase 0 benchmark: 500 virtual patients

CircAdapt physiology, domain-randomised device signals, patient-level split.

| Model | In-domain near-optimal | In-domain regret | Shifted near-optimal | Shifted regret |
|---|---:|---:|---:|---:|
| linear | 0.640 | 0.0180 | 0.640 | 0.0217 |
| linear_no_sensors | 0.493 | 0.0366 | 0.493 | 0.0366 |
| cnn | 0.107 | 0.1561 | 0.107 | 0.1467 |

The shifted domain is a synthetic stress test (noisier signals, different
impedance calibration, broader EGM wavelets), not real patient data.

CNN: about 2412 parameters, best epoch 12.

## Linear sensor ablations

| Feature set | In-domain near-optimal | In-domain regret | Shifted near-optimal | Shifted regret |
|---|---:|---:|---:|---:|
| all | 0.640 | 0.0180 | 0.640 | 0.0217 |
| impedance_only | 0.640 | 0.0180 | 0.640 | 0.0217 |
| no_sensors | 0.493 | 0.0366 | 0.493 | 0.0366 |
| setting_only | 0.467 | 0.0389 | 0.467 | 0.0389 |

## Label-surface diagnostics

AV grid: 80-300 ms; optimum at max: 1.3%. VV grid: -60-120 ms; optimum at max: 0.0%.

This demonstrates *learnability in simulation* and robustness to a
controlled domain shift. It is not clinical validation.
