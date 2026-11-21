# Pre-registration of the holdout evaluation (written 2026-11-21, before any holdout metric was computed)

All design choices were made on the development period (out-of-sample years 2016–2022) and are
recorded in `RESEARCH_LOG.md` (iterations 0–6). The holdout (out-of-sample years 2023–2025, yearly
walk-forward refits, point-in-time S&P 500) is evaluated **once**, for the systems below, and every
result is reported whatever it shows. The evaluation is logged in `results_v2/summary/UNSEAL_LOG.txt`.

## Primary system (h = 21, point-in-time S&P 500)
**System A**: cross-sectional ranks of five inputs (XGBoost, Ridge and MLP walk-forward forecasts,
21-day reversal, 5-day reversal), averaged with equal weights; long-short portfolio with buffer:
enter the top/bottom 20 %, keep while within the top/bottom 50 %; 21-day holding; costs per side
on traded notional; all statistics averaged over the 21 rebalancing phases.

Primary hypotheses (one test each, α = 0.05):
* H1: mean daily rank IC of the System A signal > 0 (Newey–West t-test, lag 20, one-sided).
* H2: net long-short Sharpe ratio at 5 bp per side > 0 (reported with the gross, 10 bp and 20 bp
  values, turnover and maximum drawdown).

## Secondary results (reported, no claims beyond description)
* S1: System A at h = 5 (same inputs; 5-day holding).
* S2: the full system (System A + walk-forward regime exposure gate) at h = 21 and h = 5.
* S3: System B (reversal components gated by their own regime gate) at h = 21.
* S4: every base model at h = 21 and h = 5 (XGBoost, Ridge, MLP, XGBoost-IC; Transformer, AFT and AFT-R at
  h = 21), plus the classic factors; AFT vs Transformer IC difference (Newey–West).
* S5: replication of the regime dependence of 5-day reversal (IC by tercile of VIX, market volatility,
  cross-sectional dispersion, market 63-day return and breadth; HAC slope p-values).

## Fixed before unsealing
* No hyper-parameter, input, rule or threshold of any system above is changed after this file was
  written.
* Known limitations stated in advance: the universe is today's S&P 500 members, each from its
  index-addition date (stocks deleted before 2026 are missing → survivorship bias); Yahoo Finance
  adjusted prices; no borrowing costs; equal-weight legs; and the holdout has only about three
  years (≈ 750 dates, ≈ 36 non-overlapping 21-day periods per phase), so power is limited.
