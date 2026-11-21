# Audit of the AFT manuscript, notebooks and benchmark outputs

Scope: `research.tex` (rejected conference version), `paper.tex` / `2606.29347v1.pdf` (arXiv v1),
`README.md`, `improvement.md` (reviews), notebooks `00`–`14`, and every file in `experiments/`.
Every claim below was checked against notebook code and printed outputs, or recomputed from the saved CSVs.
"NBxx" = notebook `xx_*.ipynb`; "cell n" = notebook cell index.

Severity: **S1** = changes a reported result or a conclusion, or is a number that is not produced by any
experiment; **S2** = a method is described incorrectly; **S3** = presentation or minor inconsistency.

---

## A. Numbers in the manuscript that no experiment produced (S1)

| # | Manuscript claim | What the code and outputs show |
|---|---|---|
| A1 | Table II / arXiv Tables IV–V: ±95 % CIs for Linear Regression, Random Forest, XGBoost, LSTM, GRU (e.g. LR MAE 0.1887 ± 0.0450). | Each baseline was trained **once** (NB03, seed 42). No notebook computes a CI for them (`baseline_results.csv`, `trading_results.csv` hold single values), so the ± values were not produced by any experiment. |
| A2 | GRU row: DA 44.13 %, Sharpe −0.800, return −65.16 % (arXiv also gives MDD −85.05 %). | These are the **vanilla PyTorch Transformer's** metrics (`experiment_log.csv`: 0.44126 / −0.79961 / −0.65158 / −0.85052). GRU's own output (NB03 cell 55) is DA 57.88 %, Sharpe 2.789, return +853.6 %, produced by the old backtest that compounds overlapping returns. |
| A3 | §IX ablation: −Gating (MAE 0.0348, DA 49.80 %, SR −0.082), −Regime (0.0355, 48.24 %, −0.124), −Financial context (0.0368, 46.50 %, −0.393). | None of these numbers appears in any notebook output or CSV. The ablations that were actually run are `ablation_results.csv` (NB10: 1 seed, different loss, lr and budget, plus a checkpoint bug, see C7) and `verification_results.csv` (NB10.1: 5 seeds, only "−regime" and "top-40"). |
| A4 | "+All 95 features yields MAE 0.0328, DA 51.23 %, SR 0.142" (presented as an ablation of the optimised model). | These are the Baseline AFT results, which also differ in L, d_model, lr, wd and dropout. This is not a one-factor ablation. |
| A5 | Multi-stock "Std Dev" row: R² 0.164, DA 6.88 %, SR 1.229, return 32.75 %. | Recomputed from `multi_stock_results.csv` (sample std): R² 0.168, DA 7.01 %, SR 1.221, return 32.46 %. Only the MAE std (0.0159) and the means are correct. The README repeats 6.88 % and 1.23. |
| A6 | "The best configuration (Trial 34)". | The best trial is **trial 6** (value −4.2377, `optuna_trials_financial.csv`). 34 is the number of COMPLETE trials. |
| A7 | Hardware and software: "Intel Xeon workstation, NVIDIA RTX 4090, PyTorch 2.1, Optuna 3.5"; arXiv: "≈45 min on an RTX 4090". | The notebooks ran on an **NVIDIA GeForce RTX 4060 Laptop GPU** with torch **2.11.0+cu128**, scikit-learn 1.9.0 and xgboost 3.2.0 (`requirements.txt`, NB03 cell 3). The project environment has Optuna **4.9.0**, which is missing from `requirements.txt` (that file is also UTF-16 encoded). The financial Optuna study took about 9 min 17 s (timestamps 22:34:57–22:44:14 in `optuna_trials_financial.csv`). |
| A8 | arXiv: "Integrated Gradients were computed using the Captum framework". | NB11 cell 10 uses a hand-written IG loop. Captum is not installed. |

## B. Data and evaluation protocol

| # | Sev. | Issue | Evidence |
|---|---|---|---|
| B1 | S1 | **Dataset misreported.** Paper: AAPL 2 Jan 2018 – 31 Dec 2024, 1,762 days, split 1,233/264/265. Actual main dataset: AAPL downloaded 2015-01-01 to 2025-12-31 (2,765 rows), **2,721 feature rows (2015-03-02 to 2025-12-22)**, split **1,904/408/409** rows (1,844/348/349 windows at L = 60). The 2018–2024 figure belongs to the multi-stock notebook (NB14: 1,760 raw rows, 1,725 feature rows). | NB01 cells 2, 70; NB02 cell 14; NB14 cell 6 |
| B2 | S1 | **The claimed sequence realignment is not used in the main results.** The paper (§IV-B, contribution 2) says windows `X_{t−L+1:t}` predict `y_t`. NB02, NB10.1, NB12.1 and NB13 build `X[i:i+L] → y[i+L]`, so the window ends one day **before** the forecast origin. Only NB08–NB10 and NB14 use the aligned version (`y[i+L−1]`). The main benchmark and the multi-stock study therefore use different conventions. | NB13 cell 5, NB12.1 cell 4, NB14 cell 4 |
| B3 | S1 | **Optimised and baseline AFT are not evaluated on the same test dates.** Windows are built inside each split, so a model with L = 80 loses 20 more test days than one with L = 60 (329 vs 349 samples: 2024-08-30 vs 2024-08-02 onward). The "paired" comparison pairs seeds, not identical samples. | NB13 cells 5–6; DA × n is an integer only for n = 349 / 329 |
| B4 | S2 | **No purging.** The last 5 training targets are realised with validation-period prices, and the last 5 validation targets with test-period prices. | NB02 cells 10–11 |
| B5 | S1 | **Test information in the hyper-parameter objective.** The normalisers MAE_base = 0.0316, DA_base = 0.5284, SR_base = 0.2423 are the NB10.1 **test-set** means. They set the relative weights of the three terms, so test data influenced model selection. Because the Sharpe ratio is divided by 0.2423, the score is ≈95 % driven by the validation Sharpe ratio: the coefficient per unit Sharpe is 1.24, versus ≈0.03 for a 5-point change in DA. | NB12.1 cells 0, 7; `verification_results.csv` |
| B6 | S2 | **HPO described incorrectly.** Paper search space: lr [1e−5, 1e−2], wd [1e−6, 1e−1], dropout [0, 0.3], d_model {32…160}, heads {2, 4, 8}, layers {1, 2, 3}. Code: lr [1e−5, **5e−3**], wd [1e−6, **1e−2**], dropout [0, **0.4**], d_model {**64**, 96, 128, 160, **192**}, heads {**4, 8**}, layers {**2, 3**}. The paper does not mention the 3-fold walk-forward CV the objective actually uses. The pruner is `MedianPruner(n_warmup_steps=1)` applied **per fold**, not "after 5 epochs". The TPE sampler is unseeded, so the search cannot be reproduced. | NB12.1 cells 7–8 |
| B7 | S2 | Top-40 features for the HPO were chosen by a random forest fitted on rows 0–1903, which **include the HPO validation folds** (rows 475–577, 1155–1257, 1835–1937). | NB12.1 cell 3 |
| B8 | S1 | **The training loss is misreported.** §VI presents a composite loss `αMSE + βCorr + γSign` and "AdamW with cosine learning-rate scheduling". Every final model (NB10.1, 12.1, 13, 14) is trained with **`nn.MSELoss`** and **`ReduceLROnPlateau`**. α, β, γ are never defined (Reviewer B #6). In NB09/NB10, where the composite loss was used, α = 1, β = 0.005, γ = 0.002, and the sign term is −mean(sign(y)·tanh(10ŷ)), not the hinge of Eq. 20. The symbol γ is also reused for the context scale σ(γ). The abstract instead describes the *Optuna score* (MAE, DA, Sharpe) as the "composite objective". | NB13 cell 6; NB09 cells near "MultiObjectiveLoss" |
| B9 | S1 | **Transaction costs are not applied.** The paper says 0.05 % is charged on every position change, and arXiv Tables V and VIII are captioned "Transaction Cost = 0.05 %". The trading metrics of NB13 and NB14 contain **no** costs. Only NB10's function had a cost term. | NB13 cell 5; NB14 cell 4; NB10 cell 6 |
| B10 | S2 | **The Sharpe ratio is described incorrectly.** The paper says μ, σ come from *non-overlapping* returns. The code takes mean/std over all overlapping, daily-sampled 5-day returns (× √(252/5)). Strategy return and MDD use one non-overlapping phase (`[::5]`, offset 0), so the result depends on an arbitrary start day. | NB13 cell 5 |
| B11 | S1 | **Baselines use a different pipeline and an old backtest.** LR/RF/XGB/LSTM/GRU (NB03) use the 1-day-gap windows, flattened 60 × 95 inputs for the tabular models, and the old metric (compounding *overlapping* 5-day returns daily, √252 annualisation, no costs). That metric produced +853 % for the GRU. Their trading metrics cannot be compared with the AFT rows. | NB03 cells 7, 52 |
| B12 | S1 | **No naive benchmarks.** On the NB13 test window, the **zero forecast has MAE 0.0308**, better than both AFT models (0.0328 and 0.0336). **Always-long has DA 56.4 %** (the share of positive 5-day returns), versus 51.2 % / 51.7 % for the AFT models. The negative R² values already imply this, but the paper never states it. | recomputed from `master_dataset.parquet` |

## C. Other notebook and CSV defects

| # | Sev. | Issue |
|---|---|---|
| C1 | S1 | Multi-stock study (NB14) uses **undocumented hyper-parameters** (lr 3.57e−4, wd 4.17e−5, dropout 0.03, L = 60, d_model 160, d_ff 320). These are neither the baseline nor the Optuna result. It also uses **different feature engineering** (≥10 of 95 features differ from NB01: Price_Change, ROC × 100, 30-day historical volatility on simple returns, Parkinson on price ranges, 30-day Garman–Klass, EMA `adjust=False`, skew/kurtosis of prices, Distance_From_* as differences, Volume in the price group), a 15-epoch budget, patience 4, no gradient clipping, default AdamW betas, aligned targets, no costs, and one seed. |
| C2 | S1 | AAPL discrepancy explanation (§VIII). The text announces "three key factors" but lists two, and it invokes "cross-validation fold averaging", which neither experiment performs. The actual differences are: data period (2018–2024 vs 2015–2025), and therefore the test window (**2024-03-08 → 2024-12-19** vs **2024-08-02 → 2025-12-22**); hyper-parameters, features, alignment and training budget (C1); and one seed vs five. |
| C3 | S1 | `experiment_log.csv`: baseline trading columns are **attached to the wrong models**. NB03 cell 62 pairs `results` rows (LR, RF, LSTM, GRU, XGB) with `trading_results` rows (LR, RF, XGB, LSTM, GRU). |
| C4 | S1 | `final_rankings.csv` is hard-coded: "Regression Rank 1 … with smaller prediction error" for the optimised model, whose MAE and RMSE are *higher*. The README repeats the claim. |
| C5 | S3 | `ablation_results.csv`, "Walk-Forward CV (Average)": RMSE = 0, DA = 0.5, Sharpe = 0 are placeholders (NB10 cell 18 comments say so). |
| C6 | S2 | NB13 `final_benchmark_results.csv` correctly computes paired t-tests, t-based CIs and d_z. I recomputed all of them (✓). Its "Cohen's d" is d_z (mean difference / SD of differences) and should be labelled that way. |
| C7 | S2 | NB10 `best_model_state = model.state_dict().copy()` is a *shallow* copy. The "best" checkpoint is therefore the last epoch's weights, so every NB10 ablation number is affected. |
| C8 | S3 | NB13 inference time includes CUDA warm-up. Train time depends on early stopping, so the timing comparison is not a controlled measurement. |

## D. Architecture and method description (S2)

| # | Manuscript | Code (NB12.1 / NB13) |
|---|---|---|
| D1 | Regime encoder is "GRU + Pooling" (Fig. 1, §XII "GRU-based"); Eq. 10 is `Pooling(GELU(X_g W_g + b_g))`. | No GRU. Per group: `Linear(D_g, 64) → GELU → Linear(64, 32)` at every time step, then **learned attention pooling** over time (`softmax(Linear(32, 1))`) and fusion `Linear(32G, 128) → GELU → Dropout → Linear(128, 32)` (no final GELU). |
| D2 | Gate: `softmax(R W + b)`. | `Linear(32, 64) → GELU → Dropout(0.15) → Linear(64, G) → softmax`. |
| D3 | `B_fin = σ(γ) Σ_g w_g M_g`. | `B_fin = σ(φ) Σ_g softmax(π)_g · w_g · M_g`, with an additional learnable static group weight π. The temporal bias is also scaled, by σ(ψ). M_g is computed from the raw (standardised) inputs with a zero row prepended for the CLS position. |
| D4 | Algorithm 1, line 11: `LayerNorm(Z + FFN(Z_out))`. | Pre-norm residual blocks: `x ← x + Drop(Attn(LN(x)))`, then `x ← x + Drop(FFN(LN(x)))`. Head: `LN → Linear(d, 64) → GELU → Dropout → Linear(64, 1)` on the CLS token. |
| D5 | Context cost O(G L²). | O(G L² d_head) for the similarity products, plus O(L D d_head) for the projections. |
| D6 | **Gate attenuation (new finding).** | Because the static weights softmax(π) and the gates w both lie on the simplex, Σ_g softmax(π)_g w_g ≈ 1/G ≈ 0.09. With σ(φ) = σ(0.5) = 0.62, the whole context bias is about 0.056 × (mean similarity), and a gate change of ±0.01 moves the attention logits by about 10⁻⁴. This explains D7. |
| D7 | Explainability: "during low-volatility regimes price and trend dominate; during high-volatility regimes volatility and breakout features get higher gate probabilities". | NB11 cell 7 output: every gate lies in **0.080–0.0955** (uniform = 0.0909), and the three K-means clusters differ only in the third decimal. The legend labels "Low Vol / Trending", "High Vol / Crash" and "Mean Reverting" are typed into the plotting code, not derived from data. |
| D8 | IG figure presented as general feature importance; temporal plot shows "highest importance to recent events". | IG is computed for **one** test sample (the largest prediction) of the *baseline* checkpoint `models/adaptive_financial_transformer.pth`, not of the optimised model. The x-axis says "Days Ago", but index 1 is the *oldest* step. |

## E. Claims not supported by the evidence (S1)

* "Optimised AFT … improves directional accuracy (51.23 → 51.67 %) and Sharpe (0.142 → 0.161)": p = 0.90 and 0.95. The changes are not significant, both values are below the always-long DA (56.4 %), and both MAEs are worse than the zero forecast (B12).
* "Domain-specific structural priors improve parameter and sample efficiency" (Conclusion): no experiment varies sample size, and none compares against a parameter-matched control.
* "Top-40 … lowering validation variance across rolling windows": the only walk-forward run (NB10) is a placeholder row (C5).
* "Multi-stock evaluation validates predictive generalizability": it covers 1 seed and an undocumented configuration (C1). Mean DA is 53.4 % with no comparison to always-long or to any baseline.
* "Gated attention parameters saturate" on NVDA: never measured. Saturation is also inconsistent with D6/D7 (the gates are near-uniform, not saturated).
* README "slightly better Sharpe ratios", "validating its predictive generalizability", "Regression Rank 1": same issues.

## F. Verified as correct

* Paired t-test p-values, t-based 95 % CIs (t₀.₉₇₅,₄ = 2.776) and d_z in `final_benchmark_results.csv` / Table II for the two AFT models.
* Parameter counts: 373,143 (baseline) and 316,319 (optimised). Reproduced exactly by `aft/models.py`. Reductions: 15.23 % parameters, 57.9 % features (55 of 95).
* Multi-stock means (MAE 0.0353, R² −0.151, DA 53.43 %, SR 0.500, return 7.83 %).
* Optuna best hyper-parameters in `best_hyperparams_financial.json` = trial 6.
* No look-ahead in the 95 engineered features themselves. Every feature at row t uses data up to the close of t; the scaler is fit on training rows only.
* The "duplicated 58 %" formatting issue reported by Reviewer B is **not** present in the LaTeX source of Table II (the RF row reads `51.58\%`).

## G. What was done about it

All results in the revised manuscript were regenerated by a single reproducible pipeline (`aft/`,
`scripts/`) that fixes B1–B12 and C1–C8:

* aligned windows; purged boundaries; **identical forecast dates for every model** (windows may draw
  features from earlier splits);
* identical training protocol and **5 seeds for every stochastic model**, including all baselines;
  deterministic CUDA kernels (re-running reproduces every prediction bit for bit);
* naive benchmarks (zero, train mean = buy & hold, 5-day momentum), tabular models on the current
  feature vector (Ridge, RF, XGBoost), sequence models (LSTM, GRU, DLinear, vanilla Transformer
  with the AFT's embedding, depth and head);
* cost-aware non-overlapping backtest, averaged over all 5 rebalancing phases; costs of 0/5/10/20 bp
  per side; turnover reported;
* Diebold–Mariano (HLN-corrected) tests on seed-ensemble forecasts; HAC tests of DA; paired seed tests
  with d_z;
* a leakage-free re-run of the HPO (validation-only objective, per-fold feature selection, seeded TPE);
* 5-seed, one-factor ablations with identical initialisation of shared weights;
* an 11-asset study (6 tech, 4 other sectors, SPY) with the same features and protocol, plus
  covariate-shift diagnostics;
* gate and attention-bias diagnostics computed from the saved checkpoints.

Every number in the revised paper is read from `results/` by `scripts/analyze.py`, which writes the
LaTeX tables and `numbers.tex`. No number is typed by hand.
