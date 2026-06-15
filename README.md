#  Lithium-ion Battery Life Prediction

**From early-prediction tabular regression to SOH time-series forecasting**

An empirical study on the standard **Severson/Attia dataset (MIT–Stanford, *Nature Energy* 2019)** —
140 LFP/graphite cells. The project investigates **two complementary problems** on the same data source:
*early* prediction of battery cycle life (tabular) and *sequence* forecasting of battery State of Health (SOH).

> Project for **AI VIET NAM** · Team Xì Trum.

---

##  Two problems

| | Problem | Type | Input → Output |
|---|---|---|---|
| **A** | Early `cycle_life` prediction | Tabular regression | 12 features from the first 100 cycles → total cycles to end-of-life |
| **B** | SOH sequence forecasting | Time-series seq2seq | window of `L=30` past SOH values → next `H` SOH values |

**Research questions**
- **RQ1 (A)** — Can cycle life be predicted from only the early cycles? How many cycles are needed (20/50/100)?
- **RQ2 (A)** — Does the model generalize to **unseen charging policies**?
- **RQ3 (B)** — Can the SOH trajectory be forecast, and how does the error scale with the horizon `H`?
- **RQ4 (B)** — Do physical channels (internal resistance, temperature) and strong sequence models (LSTM, Transformer) beat simple baselines?

---

## 🏆 Key results

### Problem A — Early prediction (R² vs. number of early cycles, repeated CV)

| Model | N=20 | N=50 | N=100 | MAPE@100 |
|---|---|---|---|---|
| **XGBoost** | 0.634 | 0.702 | **0.706** | **16.1%** |
| RandomForest | 0.629 | 0.670 | 0.669 | 17.4% |
| ElasticNet | 0.518 | 0.704 | 0.208 ⚠️ | 18.0% |

-  Cycle life **is predictable** from early cycles (XGBoost R²≈0.71).
-  Accuracy **jumps sharply 20→50 cycles, then plateaus** → ~50 cycles already carry most of the signal.
-  Generalizing to **unseen policies**: R² drops from 0.706 → 0.626 (**ΔR² = −0.08**).

### Problem B — SOH forecasting (RMSE ×10⁻², held-out cells)

| H | Persistence | Linear | Ridge | RandomForest | MLP | LSTM | Transformer |
|---|---|---|---|---|---|---|---|
| 10 | 0.237 | 0.068 | 0.046 | 0.038 | **0.036** | — | — |
| 20 | 0.440 | 0.110 | 0.072 | 0.060 | **0.056** | 0.057 | 0.066 |
| 50 | 0.956 | 0.295 | 0.178 | **0.145** | 0.141 | — | — |
| 100 | 1.533 | 0.656 | 0.388 | **0.346** | 0.354 | — | — |

-  Error grows steadily with the horizon `H` (the core trait of forecasting).
- **Delta-modeling is the key trick**: without it, LSTM (1.113) is **worse than persistence**; with it, LSTM reaches 0.057, on par with MLP.
-  All learned models **cluster within 0.056–0.072** — the **Transformer does NOT beat** the simple baselines.

> **Honest takeaway:** on *cycle-level scalar* data, complex sequence models do not outperform simple ones —
> **the limit lies in the data, not the model**. Exploiting LSTM/Transformer requires *intra-cycle* signals
> (V/I/T / Q(V)).

---

## 📂 Structure

```
.
├── early-battery-cycle-life.ipynb    # Problem A — tabular early prediction
└── battery-soh-forecasting.ipynb     # Problem B — SOH time-series forecasting (Kaggle GPU)
```

**Notebook A** (`early-battery-cycle-life.ipynb`): EDA · feature engineering (12 features) ·
ElasticNet/RF/XGBoost · 20/50/100-cycle ablation · SHAP · Quantile RF (prediction intervals) · cross-policy (GroupKFold).

**Notebook B** (`battery-soh-forecasting.ipynb`): sliding windows · delta-modeling · baselines (Persistence/Linear) ·
Ridge/RF/MLP · LSTM encoder–decoder & Transformer · recursive full-curve forecasting (knee point).

---

## ▶️ How to run (Kaggle)

1. **New Notebook** → right panel **Add Input** → search for
   `solitaryseeker/lithium-ion-battery-cycle-life-time-series-dataset` → **Add**.
2. **File → Upload Notebook** → choose the notebook you want to run.
3. For Problem B: **Settings ⚙ → Accelerator → GPU T4** (needed for LSTM/Transformer).
4. **Run All**. The notebook auto-detects the data path under `/kaggle/input/...`, no manual edits needed.

---

##  Data

- **Source:** Severson/Attia (MIT–Stanford, *Nature Energy* 2019) — 140 LFP/graphite cells
  (A123 APR18650M1A, 1.1 Ah), ~70 charging policies, cycle lives 148–1,935 (mean 802 ± 332).
- **Each row = one cycle:** `QD` (discharge capacity), `IR` (internal resistance), `Tavg/Tmin/Tmax`,
  `chargetime`, `cycle`, `battery_id`, `cycle_life`, charging policy `C1/Q1/C2`. **SOH = QD / 1.1**, EOL at SOH = 0.8.
- **Kaggle:** <https://www.kaggle.com/datasets/solitaryseeker/lithium-ion-battery-cycle-life-time-series-dataset>

>  The dataset ships 3 overlapping CSVs (full / 50-cycle / 100-cycle) — **the same 140 cells**, differing only
> in the cycle range. Problem A uses the 100-cycle file; Problem B uses the full file. **Never `pd.concat` all three**
> (it duplicates the data).

---

##  Method (in brief)

- **Problem A:** 12 features per cell (strongest: `log10|ΔQ|`), regression on `log10(cycle_life)`;
  evaluated with **repeated 5-fold CV**, scaling + log target **fit inside each fold** → no data leakage.
- **Problem B:** sliding windows (`L=30`, stride 10), **cell-wise** split (112 train / 28 test);
  **delta-modeling** (forecast the change, then add it back) — essential for sequence models to be competitive.

---

##  References

1. Severson et al. (2019). *Data-driven prediction of battery cycle life before capacity degradation.* **Nature Energy** 4, 383–391.
2. Pham et al. (2022). *A Multitask Data-Driven Model for Battery RUL Prediction* (Conv2DLSTM + multitask).
3. Nicolae et al. (2025). *Optimizing Cycle Life Prediction via a Physics-Informed Model.* arXiv:2404.17174.
4. Singh et al. (2021). *Data Driven Prediction of Battery Cycle Life Before Capacity Degradation.* arXiv:2110.09687.

---

##  Team Xì Trum

Phan Văn Tiến · Lê Xuân Dũng · Đồng Tiến Đoàn · Võ Ngọc Gia Bảo · Phan Tâm Anh

*AI VIET NAM — 2026* · <https://aivietnam.edu.vn>
