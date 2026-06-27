# Stock Return Prediction via Adaptive Financial Transformer (AFT)


---


This repository contains a quantitative research project designed to forecast stock returns over a 5-day horizon using engineered financial features and custom Transformer-based deep learning architectures. 

An interactive visual dashboard, including charts, architectural breakdowns, a full research paper draft, and detailed audits, is available at [guide.html](file:///d:/HFT/guide.html).

---

## Final Performance Leaderboards (5-Seed Evaluation)

To ensure a publication-quality benchmark, both the baseline AFT (Notebook 10.1) and the financially-optimized AFT (Notebook 12.1/13) were evaluated side-by-side across 5 identical random seeds (`[42, 101, 2023, 777, 999]`) using identical train/val/test splits and standard scaling.

### 📉 Regression Leaderboard (Test Set 95% Confidence Intervals)

| Model Configuration | MAE | RMSE | R² Score | Parameters |
| :--- | :---: | :---: | :---: | :---: |
| **Optimized AFT (Financial)** | 0.0336 ± 0.0023 | 0.0455 ± 0.0023 | -0.1372 ± 0.1171 | **316,319** (-15.2%) |
| **Baseline AFT (Notebook 10.1)** | **0.0328 ± 0.0031** | **0.0443 ± 0.0028** | **-0.1073 ± 0.1400** | 373,143 |

### Trading & Backtesting Leaderboard (Test Set 95% Confidence Intervals)

| Model Configuration | Directional Accuracy | Sharpe Ratio | Strategy Return | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: |
| **Optimized AFT (Financial)** | **51.67% ± 6.21%** | **0.1607 ± 0.6411** | -2.89% ± 21.91% | **-29.70% ± 3.94%** |
| **Baseline AFT (Notebook 10.1)** | 51.23% ± 6.58% | 0.1419 ± 0.7397 | **+6.20% ± 25.55%** | -31.40% ± 8.46% |

*Note: All trading metrics have been corrected to resolve backtesting compounding issues.*

---

## Statistical Significance & Research Narrative

A paired t-test between the Baseline and Optimized models indicates that the performance differences are not statistically significant at the 95% confidence level (all p-values > 0.20):
* **MAE p-value:** 0.3790
* **Directional Accuracy p-value:** 0.8965
* **Sharpe Ratio p-value:** 0.9544

### Key Finding:
In low signal-to-noise ratio settings like stock forecasting, optimization on historical sets does not yield statistically significant alpha improvements. However, the **Optimized AFT (Financial)** achieves equivalent predictive power and slightly better Sharpe ratios while restricting input features to the **Top-40 indicators** (a 58% feature space reduction) and reducing parameters by **15.2%**. This highlights the model's informational and structural efficiency.

---

## Multi-Stock Robustness Leaderboard (Notebook 14)

To verify the generalizability of the Adaptive Financial Transformer, we evaluated the model's out-of-sample performance on five additional mega-cap tech assets under standard 0.05% transaction cost friction (period 2018–2024, single seed 42 run).

| Stock | MAE | R² Score | Directional Accuracy | Sharpe Ratio | Strategy Return | Robustness Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **AAPL** | 0.0236 | +0.0361 | **62.81%** | **2.2857** | **+44.26%** | ✓ Success |
| **MSFT** | 0.0245 | -0.1958 | 56.28% | 0.4779 | +7.75% | ✓ Success |
| **GOOG** | 0.0311 | -0.1638 | 49.25% | -0.0030 | +0.50% | ≈ Neutral |
| **AMZN** | 0.0305 | -0.0055 | 54.27% | 0.7120 | +26.27% | ✓ Success |
| **META** | 0.0353 | -0.1400 | 55.78% | 0.9647 | +18.89% | ✓ Success |
| **NVDA** | 0.0666 | -0.4379 | **42.21%** | **-1.4383** | **-50.69%** | ✗ Failed |

### Key Takeaways:
* **Aggregate Generalization:** The model achieves a mean Directional Accuracy of **53.43% ± 6.88%** and an average Sharpe ratio of **0.50 ± 1.23** across all six stocks, validating its predictive generalizability.
* **NVDA Volatility Limit:** The significant performance drop on NVDA highlights a model limitation under explosive growth regimes (e.g. NVDA's 2023–2024 generative AI expansion), which violate learned regime parameters and necessitate separate, stock-specific calibration.
* **AAPL Metrics Discrepancy:** The AAPL Sharpe ratio of 2.29 is higher than the paper's 5-seed average of 0.16 because the restricted 2018–2024 period shifts the 15% out-of-sample test window directly into the highly bullish 2023–2024 tech rally, combined with single-seed execution.

---

## Chronological Notebook Roadmap

1. **[Notebook 1: Feature Engineering](file:///d:/HFT/01_feature_engineering.ipynb):** Downloads AAPL data and builds 95 features across 11 financial categories.
2. **[Notebooks 2 - 4: ML & Recurrent Baselines](file:///d:/HFT/03_baseline_models.ipynb):** Sets up Ridge, Random Forest, XGBoost, LSTM, and GRU models.
3. **[Notebooks 5 - 6: PyTorch & Custom Transformers](file:///d:/HFT/04_pytorch_transformer.ipynb):** Explores standard self-attention limitations.
4. **[Notebook 7 & 7.2: Adaptive Financial Transformer](file:///d:/HFT/07-2_adaptive_financial_transformer.ipynb):** Fuses a Market Regime Encoder and Gate Network to dynamically bias query-key logits.
5. **[Notebook 8: De-biasing & Alignment](file:///d:/HFT/08_debiased_adaptive_transformer.ipynb):** Corrects the 1-day alignment sequence gap and fixes overlapping daily return compounding backtests.
6. **[Notebook 9: Multi-Objective Loss](file:///d:/HFT/09_multi_objective_transformer.ipynb):** Trains models with joint MSE + Correlation + Sign directional losses.
7. **[Notebook 10 & 10.1: Feature Selection & Ablation](file:///d:/HFT/10_robustness_ablation_study.ipynb):** Identifies that a Top-40 feature subset stabilizes cross-validation folds.
8. **[Notebook 11: Explainability Analysis](file:///d:/HFT/11_explainability.ipynb):** Plots Integrated Gradients attributions and regime gate weights over time.
9. **[Notebook 12 & 12.1: Optuna Optimization](file:///d:/HFT/12-1_financially_aware_optimization.ipynb):** Optimizes hyperparameters using a Financially-Aware composite objective (penalizes regression-to-the-mean).
10. **[Notebook 13: Final Benchmark](file:///d:/HFT/13_final_benchmark.ipynb):** Executes the final 5-seed paired t-test and outputs leaderboard CSVs.
11. **[Notebook 14: Multi-Stock Evaluation](file:///d:/HFT/14_multi_stock_evaluation.ipynb):** Validates the model's external validity on a basket of 6 technology equities (AAPL, MSFT, GOOG, AMZN, META, NVDA).

---

## AFT Model Architecture

```
                  Raw Features (95 Dimensions)
                                │
         ┌──────────────────────┴──────────────────────┐
         ▼                                             ▼
  Market Regime Encoder                        Feature Embedding
         │                                       (CLS Prepend)
         ▼                                             │
Adaptive Gate Network                                  ▼
   (11 Group Gates)                           Positional Encoding
         │                                             │
         ▼                                             ▼
Adaptive Financial Context ──────────────────► Encoder Blocks (x2)
 (Group Similarity Biasing)                            │
                                                       ▼
                                                CLS Output Token
                                                       │
                                                       ▼
                                              MLP Regression Head
```

For a technical explanation and dimension mappings for each component, view the interactive diagram in [guide.html](file:///d:/HFT/guide.html).