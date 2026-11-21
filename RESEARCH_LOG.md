# Research log (v2: panel re-engineering and meta-model)

Protocol (fixed 2026-11-21, `aft2/protocol.py`): walk-forward OOS years 2016–2025 with yearly refits
(expanding window, 252-day validation, 5-day purges). **Development** = OOS years 2016–2022 (all
iterations are judged here). **Sealed holdout** = 2023–2025, evaluated once for frozen designs.

## Iteration 0 — why v1 could not work (AAPL, corrected protocol)
* On the corrected single-stock protocol, no learned model beats the training-mean (buy & hold)
  forecast. LSTM, GRU and DLinear are significantly worse than the zero forecast (DM p ≈ 0.01–0.02).
  The AFT's ensemble DA (49.1 %) is below the 57.2 % up-rate.
* Random forest MAE 5.15 % (vs 3.12 % for the zero forecast): 48 of the 95 v1 features
  (`aft.data.LEVEL_FEATURES`) are price, volume or cumulative *levels* that leave their training range,
  the same covariate shift that broke NVDA.
* AFT gates are inert: the effective group weights are a product of two softmaxes (≈ 1/G).
* v1 ablations (AAPL, 5 seeds, `results/summary/aapl_summary_test.csv`): removing the financial-context
  bias, the temporal bias, or both changes nothing (test MAE 0.0315 in all four cases; paired p ≥ 0.47
  for MAE, |ΔSharpe| ≤ 0.03). The AFT's advantage over the stock `nn.TransformerEncoder` model with the
  same parameter count (MAE 0.0315 vs 0.0341) is therefore due to the custom attention block's
  implementation, not to the financial components. AFT-R, the composite training loss, Top-40 features
  and the v1 tuned configuration do not help (the last three are worse).
* Decision: re-engineer from scratch: (1) panel of 100 stocks, (2) stationary, scale-free features with
  per-date rank normalisation, (3) cross-sectional (relative-return) target, (4) meta-model over a
  model zoo, (5) sealed holdout.

## Data and features v2
* 100 large US stocks, 11 sectors, 2010–2025 (Yahoo, adjusted). BK dropped (no data). DHR used from
  2016-07-05 (mis-adjusted Fortive spin-off).
* 63 features (48 stock-level in 9 groups + 15 market and calendar): see `aft2/features.py`.
* Look-ahead test: features recomputed on data truncated at 6 random dates are **exactly** equal to
  the full-sample features (63 features × 100 stocks).
* 373,548 valid stock-days.

## Iteration 1 — base zoo (MSE, fixed hyper-parameters) and first meta-learners
Setting: 100 stocks, h = 5, relative returns, L = 20, 3 seeds; dev OOS 2016–2022
(`results_v2/summary/top100_h5_dev_models.csv`).
* Mean daily rank IC: MLP 0.0076, Ridge 0.0070, XGBoost 0.0059, DLinear −0.0086. None is
  significant (|t| ≤ 1.5). The best single signal is the 5-day reversal factor (IC 0.0129, t = 1.48).
* Long-short quintile Sharpe ratio before costs is ≤ 0.44. With turnover of 2–3× the book per weekly rebalance, 5 bp
  per side brings every model to ≈ 0 or below.
* IC flips sign from year to year for most signals.
* Meta v1 (OOS 2017–2022; `results_v2/meta/dev_meta_v1.csv`):
  equal-weight 0.0095, skill-weighted −0.0064 (trailing skill is *anti*-persistent), stacked ridge −0.0023,
  stacked XGB −0.0071, gated stacking 0.0040. **No meta-learner beats its best input.**
* H1 (turnover smoothing by causal EMA of ranks): turnover drops 2.2 → 0.8, but IC decays, so net Sharpe
  stays ≈ 0. **Rejected.** Conclusion: the binding constraint is the amount of signal.
* Feature ablation (same panel): XGBoost on the *v1* features (pooled z-score) reaches IC 0.0137
  (t = 2.10) vs 0.0059 on v2 features; the difference is not significant (p = 0.42), and one t ≈ 2 among
  this many configurations is expected by chance. Kept as an open question for the holdout.
* Feature 2×2 completed (dev, h = 5, 100 stocks): XGBoost v1+rank IC 0.0136 (t = 2.48), v1+z 0.0137,
  v2+rank 0.0059; MLP v1+rank 0.0075, v2+rank 0.0076, v1+z 0.0044. On a development refit (train
  ≤ 2018), however, the v1 XGBoost early-stops at **iteration 0**: its ranking comes from one tree splitting
  on VWAP and OBV, which are cumulated since 2010 and have no cross-sectional economic meaning. The v1
  edge is therefore treated as spurious, and the v1 features are not carried forward. Side finding: MSE
  early stopping is a poor criterion for ranking with weak signal (it prefers the near-constant model).
* Decision: test breadth (point-in-time S&P 500) and horizon (21 days) next.

## Iteration 2 — horizon and breadth
* 21-day horizon, 100 stocks (`top100_h21_dev_models.csv`): ICs a little higher (21-day reversal 0.027,
  MLP 0.013), but none is significant once 20 lags of overlap are accounted for. Ridge keeps a net
  long-short Sharpe of 0.31 at 5 bp, because monthly rebalancing cuts trading cost about 4×. Horizon alone is not decisive.
* Point-in-time S&P 500 panel: 500 current constituents, each eligible only from its index-addition
  date (Wikipedia list, retrieved 2026-11-21); 268–484 members per date; 1.39 M valid stock-days.
  Corporate-action artefacts removed (DHR, EXPE, HPQ, DD, RTX, IFF start at the event date; see
  `aft2/universe.py`). Look-ahead test passed. Remaining bias: stocks deleted from the index before
  2026 are missing (survivorship); stated as a limitation.

## Iteration 3 — meta v2: regime-gated exposure and long-window skill (100 stocks, h = 5)
(`results_v2/meta/dev_meta_top100_h5_v2.csv`; timing evaluated on 2018–2022, walk-forward ridge of the
signal's realised daily IC on 7 market-state variables, exposure = 1 if predicted IC > 0)
* 5-day reversal: gross long-short Sharpe 0.27 → 0.49, net (5 bp) −0.10 → +0.24, mean exposure 54 %.
* MLP: 0.36 → 0.51 gross, −0.01 → +0.24 net (exposure 62 %).
* Equal-weight combination: 0.22 → 0.37 gross, −0.08 → +0.11 net (66 %).
* Ridge: 0.84 → 0.62 gross, 0.48 → 0.32 net (worse).
* Long-window (252-day) skill weighting: IC −0.004. Skill-based weighting fails at both windows.
* Interpretation: consistent with short-term reversal profits being compensation for liquidity
  provision, which is higher in stressed markets (Nagel 2012). It is the first evidence that *regime
  gating helps at the decision (exposure) level*, where it failed at the feature level (v1 AFT).
  Not yet significance-tested, and several variants were tried: it must replicate on the S&P 500 and
  then on the sealed holdout.

## Iteration 4 — breadth (point-in-time S&P 500, h = 5) and the layered meta system
Base models (`results_v2/summary/sp500_h5_dev_models.csv`, dev 2016–2022):
* XGBoost IC 0.0155 (t = 2.00, positive in all 7 years), Ridge 0.0123 (t = 2.02), MLP 0.0102 (t = 1.55);
  5-day reversal 0.0156 (t = 2.00), 21-day reversal 0.0196 (t = 2.31). **Breadth turns weak, insignificant
  signals into detectable ones** (the 100-stock ICs were 0.006–0.013 with |t| ≤ 1.5).
Meta-learners (`dev_meta_sp500_h5_v2.csv`): equal-weight ranks IC 0.0208 (t = 2.48) beat every input;
all learned combiners fail again (stacked ridge −0.002, stacked XGB 0.003, gated stacking 0.003,
skill-weighted −0.005). The forecast-combination puzzle holds in both universes. The regime gate
replicates for 5-day reversal (net Sharpe 0.11 → 0.34) but not uniformly.
Buffer rule (`sp500_h5_dev_buffer.csv`): enter top/bottom 20 %, keep within 50 %: for the combination,
net Sharpe @5 bp 0.29 → 0.37 and @10 bp −0.05 → 0.15, with turnover 2.47 → 1.29.
Layered system (`metasystem_sp500_h5_dev.csv`, 2018–2022, inputs XGBoost, Ridge, MLP, rev5, rev21):
| system | IC | SR@5 | SR@10 | turnover |
|---|---|---|---|---|
| best single input (rev21) | 0.017 | 0.30 | 0.11 | 1.56 |
| L2 EW combination | 0.020 | 0.33 | 0.03 | 2.41 |
| L2 + regime gate | 0.020 | 0.27 | 0.02 | 1.89 |
| L2 + buffer | 0.020 | 0.39 | 0.20 | 1.22 |
| full (gate + buffer) | 0.020 | 0.31 | 0.12 | 1.14 |
**Design decision (made before any holdout evaluation):** the frozen meta system is
L2 + L4 (equal-weight rank combination + buffer 20/50). The gate is retained only as a reported
variant (it helps short-term reversal alone, consistent with liquidity-provision economics, but
not the combination).

### Regime dependence of short-term reversal (S&P 500, dev 2016–2022; descriptive)
5-day reversal IC by tercile (low / mid / high) of the market state, with the HAC p-value of the slope
(lag 4): VIX 0.007 / −0.001 / 0.041 (p = 0.031); market volatility 0.006 / 0.005 / 0.036 (p = 0.001);
cross-sectional dispersion 0.004 / 0.001 / 0.043 (p = 0.001); market 63-day return 0.033 / 0.015 /
−0.001 (p = 0.004); breadth 0.041 / 0.004 / 0.003 (p = 0.008). For 21-day reversal the pattern has the
same sign but is weaker (p 0.05–0.5). This replicates the liquidity-provision interpretation of
short-term reversal (Nagel 2012) on a point-in-time S&P 500. It is where the project's original
"regime" idea carries significant information: the regime state should scale *exposure to
liquidity-provision signals*, not gate feature groups inside attention.

System B (reversal components gated by their own walk-forward regime gate, model components always
on), 2018–2022: net Sharpe @5 bp 0.28 without buffer, 0.35 with buffer (System A: 0.33 / 0.39).
**Decision:** System A = primary frozen design; System B = pre-registered secondary variant.

## Iteration 5 — 21-day horizon on the point-in-time S&P 500
Base models (`results_v2/summary/sp500_h21_dev_models.csv`, dev 2016–2022, Newey–West lag 20):
Ridge IC 0.0296 (t = 2.75), XGBoost 0.0345 (t = 2.54), MLP 0.0329 (t = 1.99); 21-day reversal 0.0374 (t = 2.68).
Momentum, low-volatility and MAX are negative in this period.
Meta system (`metasystem_sp500_h21_dev.csv`, 2018–2022):
| system | IC | SR@5 | SR@10 | SR@20 | turnover per rebalance |
|---|---|---|---|---|---|
| best single input (Ridge) | 0.036 | 0.76 | 0.64 | 0.39 | 2.55 |
| L2 EW combination | 0.040 | 0.59 | 0.48 | 0.27 | 2.86 |
| L2 + gate | 0.040 | 0.66 | 0.55 | 0.34 | 2.72 |
| **System A** (L2 + buffer) | 0.040 | 0.59 | 0.51 | 0.36 | 1.70 |
| full (gate + buffer) | 0.040 | 0.67 | 0.58 | 0.42 | 1.71 |
| System B + buffer | 0.036 | 0.53 | 0.46 | 0.31 | 1.59 |
Notes: at h = 21 the gate adds a little; at h = 5 it hurt. Ridge alone has the highest Sharpe in these years,
but it was the weakest base model at h = 5 (0.04), so picking it from this table would be selection on
the evaluation period. Decision: the primary setting for the holdout is **h = 21, S&P 500 point-in-time,
System A**. The h = 5 System A and the gated variants are pre-registered secondary results.
The architecture question (Transformer vs AFT vs AFT-R) is moved to h = 21 (stride-5 training dates).

## Iteration 6 — two more base/meta ideas (S&P 500, h = 21)
* XGBoost-IC (number of trees chosen by validation rank IC instead of MSE): IC 0.0317 vs 0.0345 for
  standard XGBoost (p = 0.67). **Not adopted.**
* Regime-tilted equal weight (EW + ridge-shrunk interactions rank_m × state_k, penalty chosen on past OOS
  data only; `dev_meta_sp500_h21_v3.csv`): IC 0.0139 vs 0.0301 for EW over the same eight inputs (p = 0.45);
  net long-short Sharpe 0.31 vs 0.25. This is the sixth learned combiner that does not beat equal weighting on IC.
  At h = 5 it is clearly worse (IC 0.0033 vs 0.0208 for EW; net Sharpe −0.11). **Not adopted.** The regime
  dependence of reversal is real (Iteration 4), but estimating it inside a
  combiner adds more noise than signal at this sample size.

## Holdout (2023–2025), evaluated once per PREREGISTRATION.md — results as observed
(`results_v2/logs/holdout.log`, `results_v2/summary/holdout_report.json`, `metasystem_sp500_h{21,5}_holdout.csv`)
**Primary (System A, h = 21): not supported.** IC 0.0003 (t = 0.02, one-sided p = 0.49; by year 2023 −0.022,
2024 −0.010, 2025 +0.036). Net long-short Sharpe @5 bp 0.08 (gross 0.18, @10 bp −0.02, @20 bp −0.22).
The 21-day ML signals reversed in the 2023–24 mega-cap rally: XGBoost IC −0.021, Transformer −0.018, AFT −0.019,
AFT-R −0.016; only Ridge stayed positive (0.018, net Sharpe 0.46), and momentum turned positive.
**Secondary S1 (System A, h = 5): positive.** IC 0.0172 (t = 1.77, one-sided p = 0.038), positive in all
three years (0.018 / 0.009 / 0.025); net Sharpe @5 bp 0.34 (gross 0.64, @10 bp 0.05), max drawdown −11 %.
S2 full system (gate + buffer) at h = 5: net Sharpe 0.62 @5 bp, 0.32 @10 bp (exposure 65 %); at h = 21: −0.33.
S4: at h = 5, XGBoost IC 0.021 (t = 2.29) and XGBoost-IC 0.025 (t = 2.35) replicate the development result;
AFT vs Transformer at h = 21: ΔIC −0.001 (p = 0.91).
S5: the regime dependence of 5-day reversal did **not** replicate (all slopes p ≥ 0.07; VIX p = 0.52).
Reading: the development choice of h = 21 as the primary setting was wrong in hindsight. The weekly (h = 5)
cross-sectional signal from tree models, combined and executed with a buffer, is the result that
survived a sealed out-of-sample test. Because S1/S2 were secondary, they are evidence, not confirmed
claims; any new design built on them needs a fresh untouched period (e.g. 2026 data as it arrives).
