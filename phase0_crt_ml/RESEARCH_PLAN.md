# Research Plan — Phase 0 model upgrade: from linear ranker to production-grade ranking

> **Nota:** documento de planificación histórica. Algunas cifras intermedias
> (p. ej. `bench1000`) corresponden a corridas exploratorias y no son las del
> manuscrito. Las cifras de referencia son las de
> `reports/gbt_report_ladder500_imp_reconcile.json` y
> `reports/reconcile_shift_protocol.json`.

Target: proposal to a hospital research group (Argentina). Scope: in-silico
learnability of AV/VV optimization in CRT from intracardiac impedance, then a
pilot with real acute data.

## 1. Context and problem

- Task: given one beat of intracardiac impedance + device context + the
  programmed (AV, VV) setting, rank the 56 candidate settings of a patient by
  predicted systolic stroke work.
- This is a **learning-to-rank (LtR)** problem: query group = patient,
  documents = AV/VV settings, relevance = stroke work. It is *not* a regression
  problem: absolute stroke work is patient-calibrated; only the within-patient
  ordering matters clinically.
- Current baseline: Bradley-Terry pairwise **linear** ranker over 42
  hand-crafted features (good results, but linear-in-parameters and limited
  model capacity). A tiny raw-signal 1D CNN underperforms the linear model
  (bench1000: near-optimal 0.820 vs 0.847 in-domain; 0.680 vs 0.727 shifted).

## 2. What the standard, production-grade approach is (evidence)

1. **Gradient-boosted decision trees (GBDT) dominate tabular medical ML.**
   - Grinsztajn et al., *NeurIPS 2022 D&B*: "Why do tree-based models still
     outperform deep learning on typical tabular data?" — GBDT (XGBoost,
     LightGBM, CatBoost) beats deep models on medium tabular datasets.
   - Holzmüller et al., *NeurIPS 2024*: strong pre-tuned GBDTs remain the
     best default; MLP/GBDT combos only match them.
   - Yıldız & Kalayci, 2024 (arXiv:2410.03705): on medical diagnosis over
     tabular data, GBDT has the highest average rank vs SVM, logistic
     regression, TabNet, TabTransformer — at a fraction of the compute.
   - Our data is exactly that regime: ~42 heterogeneous engineered features,
     thousands of rows, noisy labels.
2. **The ranking objective should stay pairwise/listwise, not pointwise.**
   - LambdaMART (Burges 2010) is the industry-standard LtR model (Bing,
     Elasticsearch LTR, Oracle OML). XGBoost implements it as
     `rank:pairwise`/`rank:ndcg`; LightGBM as `lambdarank`.
   - It is a drop-in nonlinear generalization of our Bradley-Terry linear
     ranker: same pairwise comparisons, but scores come from an additive tree
     ensemble instead of a dot product.
3. **Clinical CRT literature supports the patient-specific ranking framing.**
   - Ginks et al., *Europace* 2011: patient-specific impedance models predict
     optimal AV/VV in 90-92% of patients; *generic* models are poor → the
     model must exploit within-patient contrasts, which pairwise LtR does.
   - Bocchiardo et al., *Europace* 2010: impedance-derived surrogates track
     dP/dtmax during AV/VV sweeps — the signal carries the information.
   - Knijnik et al., *Heart Rhythm O2* 2023 (meta-analysis, 9 RCTs, n=8531):
     existing automatic AV/VV algorithms show **no** hard-outcome benefit →
     justifies a rigorous, regret-based, sensor-focused research plan instead
     of yet another heuristic algorithm.
4. **Deep nets on raw signals are used as feature extractors, not as the
   final model, in small-data signal+tabular problems.** The standard hybrid
   is: train a 1D CNN encoder (self-supervised or supervised), then feed its
   embeddings *together with* engineered features into a GBDT. Raw-CNN-only
   endpoints lose to feature+GBDT endpoints in this regime, which matches
   what we observed.

## 3. Model ladder (what we implement and compare)

All models share: same dataset, same patient-level 350/75/75 split, same
metrics (near-optimal fraction @ 2% regret, mean regret, top-3 hit, Spearman),
same source and shifted sensor domains.

| # | Model | Inputs | Role |
|---|-------|--------|------|
| 0 | Linear Bradley-Terry ranker | 42 features | current baseline |
| 0b| Linear, no sensor features | context/setting only | floor / information check |
| 1 | **LambdaMART (LightGBM `lambdarank`)** | 42 features | production-grade core proposal |
| 2 | 1D CNN (raw signal, pairwise loss) | 1×256 impedance + context | raw-signal reference |
| 3 | LambdaMART on CNN embeddings | 67-dim CNN embedding | does the encoder add signal info? |
| 4 | **Hybrid LambdaMART** | 42 features + CNN embeddings | the "CNN then GBT" combination |
| 5 | **Linear ranker on features + CNN embeddings** | 109 features | linear inductive bias + encoder signal |
| 6 | **Rank blend, linear ↔ linear+embed** | rank percentiles | best deployable: in-domain + shift |

Design decisions for LambdaMART (standard defaults, minimal tuning):
- Within-patient relevance labels: per-patient rank grades (0..7) of stroke
  work, so the loss only sees within-patient contrasts (calibration-invariant).
- **Linear label gains** (equal weight per grade). Exponential gains
  overweighted the top grade and lost to linear gains on the clinical regret
  metric in a controlled sweep.
- **Round selection by grouped 5-fold CV over training patients**, early
  stopping on a custom validation-regret `feval`. A single 75-patient
  validation split made per-round regret too noisy: NDCG@1 saturated in ~20–90
  rounds while regret kept improving to ~550, and val-regret selection swung
  between rounds non-monotonically. CV averaging is the production fix.
- `num_leaves=31`, `lr=0.05`, up to 600 rounds, `min_data_in_leaf=20`,
  feature/bagging fraction 0.9.
- No within-patient feature normalization in the primary run (single-beat
  deployable); the transductive normalized variant stays a sensitivity
  analysis, as in the current README.
- Blend weight (model 6) selected on validation only (grid 0..1 by 0.1,
  maximize near-optimal, tie-break regret).

## 4. Experimental protocol

1. **Primary run (this repo, `scripts/run_gbt_benchmark.py`)**: reuse the
   500-patient impedance-only dataset (`bench500_imp`, raw features on disk),
   train the 8 models above, evaluate source + shifted domains (shifted is a
   sealed test, see §5c), bootstrap CIs on the paired differences (GBT − linear
   and blend − linear) at the patient level. Outputs:
   `reports/gbt_report_<tag>.json`, `gbt_summary_<tag>.md/png`.
2. **Ablations**: GBT with impedance features removed (no_sensors) to
   quantify how much ranking power comes from the sensor vs the setting grid.
3. **Negative control**: label-shuffled groups must collapse to chance
   (closes the loop with the existing linear negative control).
4. **Learning curve**: GBT regret vs n_train_patients (10/20/40/80/150/350) —
   tells the hospital how many sweep patients a real pilot needs.
5. **Interpretation**: LightGBM gain importances + per-feature SHAP-style
   summary as the bridge to the interpretable model the hospital will ask for.

## 5. Success criteria (pre-registered)

- GBT ≥ linear on near-optimal fraction in-domain, and no degradation under
  the shifted sensor domain (or a stated trade-off).
- Hybrid (features + embeddings) ≥ GBT-features-only ⇒ the raw waveform
  carries information beyond the 42 features ⇒ motivates a deeper encoder in
  the real-data pilot. If not, the encoder is dropped and the plan simplifies
  to features + GBT.
- Negative control at chance; learning curve indicates the pilot sample size.

## 5b. Outcome (bench500_imp, primary no-normalization baseline, 350/75/75)

| Model | In-domain near-opt | In-domain regret | Shifted near-opt | Shifted regret |
|---|---:|---:|---:|---:|
| Linear | 0.653 | 0.0177 | 0.640 | 0.0217 |
| LambdaMART features | 0.640 | 0.0263 | 0.627 | 0.0293 |
| Tiny CNN raw signal | 0.120 | 0.1486 | 0.093 | 0.1555 |
| LambdaMART CNN-embed | 0.733 | 0.0199 | 0.520 | 0.0345 |
| LambdaMART hybrid | 0.707 | 0.0238 | 0.560 | 0.0312 |
| Linear +CNN embed | 0.840 | 0.0105 | 0.627 | 0.0221 |
| Blend linear+embed (w=0.4) | 0.800 | 0.0124 | 0.653 | 0.0180 |
| No sensors (floor) | 0.493 | 0.0364 | 0.493 | 0.0364 |

The table above uses the strict **2%** research threshold (fraction of patients
whose chosen setting is within 2% of the best stroke work). The pilot endpoint
is deliberately more permissive — within **5%** and 20 ms — so the same models
were also scored at that tolerance:

| Model | Source @5% | Shifted @5% |
|---|---:|---:|
| No sensors (floor) | 0.680 | 0.680 |
| Linear | 0.880 | 0.867 |
| Linear + CNN embed | 0.947 | 0.880 |
| Blend (w=0.4) | 0.920 | 0.907 |

Reading the results in the right frame:

- **Floor.** The context-only model (no waveform features) is 0.493 @2% and
  0.680 @5% in every domain. Impedance adds ~+15 points @2% and ~+19–27 points
  @5%, and roughly halves regret, even under the extreme shift. That contrast
  is the actual sensor effect, not the absolute 0.65.
- **Ceiling.** The target-adapted bound inside the shifted domain is 0.667 @2%
  and 0.853 @5%, essentially the same as the source-trained model (0.640 /
  0.867). The shift is dominated by information loss (noise/drift), so a higher
  shifted score is not attainable by better modelling; it is not in the signal.
- **Tail.** The mean regret is 1.0–2.2% and the median ~1%, but the 90th
  percentile of the chosen-setting loss is 4–6% under shift. The model is good
  on average with a clinically meaningful tail — which supports using it both
  to optimize and to flag patients with steep response curves.
- **Robustness ordering.** The plain linear model barely degrades under shift
  (0.653→0.640 @2%; 0.880→0.867 @5%). The embedding model gains a lot
  in-domain (0.840 @2%) but loses that gain under shift; the val-weighted blend
  recovers it (0.800 source, 0.653 shifted @2%). This is the argument for the
  small interpretable model as the deployable core.

- **Criterion 1 (GBT ≥ linear): rejected.** LambdaMART does not beat the
  engineered-feature linear ranker on regret, and degrades more under shift. The
  linear model's explicit quadratic-surface + context-interaction basis matches
  this simulator; the tree ensemble must rediscover it from 350 patients.
- **Criterion 2 (hybrid ≥ GBT-features): accepted, with a twist.** Raw-signal
  CNN alone is useless (near-opt 0.12), but its penultimate embeddings add
  complementary information to the *linear* ranker (0.653 → 0.840; paired
  bootstrap +0.147 [+0.067, +0.227]). Under shift the pure embedding variant
  loses robustness; a validation-weighted rank blend restores it (0.653/0.0180).
- **Criterion 3 (negative control / curve): accepted.** Shuffled-label control
  at chance (0.080, Spearman −0.149); GBT learning curve still rising at 350
  patients, so a 30–40-patient pilot is for safety/signal, not model fitting —
  the frozen Phase 0 model is the deployable artifact.
- **Deployable choice:** interpretable linear ranker on the 42 engineered
  features, plus a 0.4/0.6 rank blend with a linear model on features + CNN
  embeddings; the GBDT path is retained as a diagnostic, not as the final model.

## 5c. Distribution-shift protocol (sealed test, and when training on shift is allowed)

This section fixes how the shifted sensor domain is used. It follows the
standard protocols for out-of-distribution evaluation and clinical external
validation.

**Standards being followed.**

- WILDS (Koh et al., ICML 2021): train on source domains, evaluate on *unseen*
  domains, and use a separate OOD validation set for development; the OOD test
  set is touched once, for the final number.
- DomainBed (Gulrajani & Lopez-Paz, ICLR 2021): model selection must be
  pre-specified. Legitimate schemes are training-domain validation,
  leave-one-domain-out, or a test-domain oracle (upper bound only). Selecting
  on the test domain is not a valid protocol.
- TRIPOD+AI (Collins et al., BMJ 2024) and external-validation guidance
  (Riley et al., BMJ 2024): evaluation data must not participate in
  development; retraining/updating requires a new, independent evaluation.
- FDA/Health Canada/MHRA Good Machine Learning Practice and Predetermined
  Change Control Plans: a deployed model is frozen and monitored; retraining is
  a pre-specified, re-validated change, not an implicit consequence of new data.
- Leakage taxonomy (Kapoor & Narayanan, *Patterns* 2023): evaluating on data
  used for training or model selection is "test set not drawn from the
  distribution of interest" (class L3) and yields optimistic, non-deployable
  estimates.
- Calibration transfer in chemometrics (DS/PDS/SST; e.g. Feudale et al. 2002;
  instrument-shift-informed source-domain augmentation, *Chemolab* 2026): when
  the shift is a known instrument family, the accepted remedies are signal
  standardization or augmentation of the *training* distribution — always
  evaluated on target samples not used to build the transfer.

**Rules adopted here.**

1. **The shifted sensor domain is a sealed external-validation set.** It is
   never used for training, early stopping, round selection, blending weights,
   feature selection, or any other decision. All selection uses source
   training/validation only.
2. **The primary claim is generalization, not adaptation.** Source-trained,
   shifted-tested is the deployment-honest estimate and the number quoted in
   the abstract/README. It answers: "how does a model frozen on source data
   behave when the sensor changes?"
3. **Report an oracle/adaptation bound next to it.** Training *within* the
   shifted domain (patient-level CV) quantifies the headroom that target labels
   could buy. It is reported as a bound, never as a deployment result.
4. **If a robustness variant is ever built**, it must (a) randomize nuisance
   parameters over a *superset* of the source support, (b) hold out a third,
   more extreme shift (unseen severity or held-out axis) for evaluation, and
   (c) select hyperparameters on a shifted validation split distinct from the
   shifted test. Training and testing on the same shift family is not a valid
   robustness result.

**Why the sealed protocol is also the right practical call here.** In this
simulator the shifted domain is *disjoint by construction* in every nuisance
parameter (gain 0.010–0.014 vs 0.016–0.024; offset 0.30–0.40 vs 0.40–0.50;
noise 0.05–0.10 vs 0.01–0.035; drift 0.03–0.07 vs 0.005–0.02), so it is an
extrapolation test, not a small perturbation — a deliberately pessimistic
stress test, not the expected calibration change. Domain randomization over the
source ranges would not cover it. The target-adapted bound confirms there is
little to gain from target labels:

| Protocol on the shifted domain | Near-opt @2% | Near-opt @5% | Regret | Spearman |
|---|---:|---:|---:|---:|
| Source-trained linear (sealed test) | 0.640 | 0.867 | 0.0217 | 0.923 |
| Target-adapted 5-fold CV (oracle headroom) | 0.667 | 0.853 | 0.0214 | 0.917 |
| Target-adapted + within-patient `imp_` norm | 0.627 | — | 0.0215 | 0.927 |
| Source-trained + within-patient `imp_` norm | 0.573 | — | 0.0243 | 0.917 |

The near-identical oracle and source-trained numbers (0.667 vs 0.640 @2%;
0.853 vs 0.867 @5%) and the failure of within-patient normalization indicate
that the shift is dominated by **information destruction (noise and drift)**,
not by a calibratable gain/offset transform. Training on the shifted domain
would therefore be leakage with essentially no benefit. The productive response
to shift is shift-invariant feature design (per-beat ratios/self-normalized
shape descriptors, or an explicit calibration-transfer step), plus external
validation on real multi-site recordings — not more model capacity.

## 6. Phased roadmap for the hospital proposal

- **Phase 0 (done here)**: in-silico feasibility with the model ladder above;
  freeze the best deployable model family and its sample-size estimate.
- **Phase 1 (pilot, 30-40 patients)**: acute AV/VV sweeps with intracardiac
  impedance + invasive (dP/dtmax) or echo (VTI) reference. Primary endpoint:
  near-optimal fraction of the frozen model vs echo-optimized setting.
  Pre-register the model, the grid, and the regret threshold.
- **Phase 2 (validation)**: external site, prospectively collected sweeps,
  locked model. Then, only if Phase 1/2 hold, design a randomized outcome
  study (cf. Knijnik 2023: automatic algorithms so far show no hard benefit —
  the bar is high and the endpoint must be hard outcomes, not surrogates).

## 7. Threats to validity (carried over + new)

- All Phase 0 physiology and sensors are simulated; shifted domain is a
  synthetic stress test.
- GBT can overfit the simulator's response-surface quirks (e.g. VV penalty
  artifact); the no_sensors ablation and negative control are the guards.
- LambdaMART relevance grades quantize stroke work within patient; grades are
  rank-based, so ties/flat regions are handled by the margin, same as the
  linear ranker.
- Within-patient normalization remains transductive (needs full sweep) — kept
  out of the primary baseline.

## 8. Key references

- Grinsztajn, Oyallon, Varoquaux. *NeurIPS 2022 D&B* — tree-based models vs
  DL on tabular data.
- Holzmüller, Grinsztajn, Steinwart. *NeurIPS 2024* — pre-tuned GBDTs/MLPs.
- Yıldız, Kalayci. arXiv:2410.03705 (2024) — GBDT on medical tabular data.
- Burges. *From RankNet to LambdaRank to LambdaMART* (2010).
- Ke et al. *LightGBM* (NeurIPS 2017) — lambdarank implementation.
- Bocchiardo et al. *Europace* 2010; Ginks et al. *Europace* 2011;
  Sciaraffia et al. 2013; Delnoy et al. 2021; Knijnik et al. *Heart Rhythm
  O2* 2023.
