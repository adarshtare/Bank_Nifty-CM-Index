


# BankNifty Constructing Cash Market Index

Rebuild a **BankNifty‑like cash‑market index** from per‑minute **LTP** (last traded price) of constituent bank stocks, compare it with the official Bank Nifty series, and export a constructed index.

This repo contains two notebooks and a few CSV artifacts that together form a small, reproducible pipeline.

---

## 1) What this project does
- **Ingests raw minute files** for NSE cash market and builds a *wide* table `finals_ltp.csv` with **one column per bank** (AUBANK, AXISBANK, …, SBIN) and rows aligned by minute.
- **Constructs a composite index** from those columns using:
  - **Free‑float market‑cap weights** (preferred, from `weights.csv` or embedded fallback), and
  - **PCA fallback weights** (first principal component of minute **log‑returns**) when official/free‑float weights are missing.
- **Level‑aligns** the constructed series to the **official** BankNifty (`bn_official.csv`) at the **first overlapping bar** (divisor surrogate). If no official series is provided, produces a **Base‑100** index.

> **Why?** Validate replication quality, diagnose deviations (weights, alignment, missing constituents), and build a robust CM index even when you don’t have up‑to‑the‑minute official weights.

---

## 2) Repository contents
- `making_dataset.ipynb` — builds the constituent‑minute panel from raw CSVs under a folder (e.g., `data/`).
- `constructing_index.ipynb` — constructs the cash‑market index from `finals_ltp.csv`, optionally aligns to `bn_official.csv`, and writes an output index CSV.
- `making_dataset.ipynb`   — code cells for making dataset
- `finals_ltp.csv` — wide matrix of per‑minute LTPs, columns are tickers (12 banks).
- `bn_official.csv` — official BankNifty minute file (contains `ltp` among other columns).
- `cm_index_mcap_weighted.csv` — sample constructed index export (`timestamp, cm_index`).
- `cm_index_te_min.csv` — generated index

---

## 3) Data schemas (as present in the provided files)
### `finals_ltp.csv`
- **Shape:** ~27,679 × 12
- **Columns:** `AUBANK, AXISBANK, BANKBARODA, CANBK, FEDERALBNK, HDFCBANK, ICICIBANK, IDFCFIRSTB, INDUSINDBK, KOTAKBANK, PNB, SBIN`
- **Index:** implicit natural numbers (minutes). No explicit timestamp column (the notebooks treat row order as minute progression and may optionally add a `timestamp` = 1..N surrogate).

### `bn_official.csv`
- **Shape:** ~28,096 × 10
- **Columns:** `index, date, time, exchange, name, ltp, last_trade_qty, total_trade_amount, total_trade_qty, source_file`
- **Series of interest:** `ltp` for official BankNifty level.

### `cm_index_from_ltp.csv`
- **Columns:** `timestamp, cm_index`
- **Rows:** (demonstrative export of a long multi‑day run).

> **Note:** Missing LTPs at the right edge are expected on partial last minutes; the constructor code uses a light **forward fill** for single‑minute gaps.

---

## 4) Methodology

### 4.1 Cash-Market (CM) Index from LTPs

Let each stock *i* have a per-minute last traded price \( P_i(t) \) and a corresponding weight \( w_i \),  
such that the total weights sum to one:

$$
\sum_i w_i = 1
$$

The raw constructed index is given by:

$$
\text{RawCM}(t) = \sum_i w_i\, P_i(t)
$$

---

### Level Alignment

If an official index series \( B_{\text{off}}(t) \) is available, align the levels at the first common time \( t_0 \):

$$
\text{CM}(t) = \text{RawCM}(t) \times 
\frac{B_{\text{off}}(t_0)}{\text{RawCM}(t_0)}
$$

If there is no official series, produce a **Base-100** version:

$$
\text{CM}(t) = 100 \times 
\frac{\text{RawCM}(t)}{\text{RawCM}(t_0)}
$$

---

**Intuition:**  
- ( P_i(t) ): Price of stock *i* at minute *t*.  
- ( w_i ): Stock’s relative importance or free-float weight.  
- ( RawCM}(t) ): Weighted sum representing synthetic index movement.  
- Level alignment ensures the constructed series sits on the same numerical scale as the official BankNifty.


## 4.2 Ridge Regularization

We estimate the weights **w** that make our constructed index best replicate the official BankNifty.

The optimization problem is:

$$
\min_{w \ge 0,\; \sum_i w_i = 1}
\sum_{t \in \mathcal{T}}
\left( I(t) - \sum_i w_i\, P_i(t) \right)^2
$$

Where:  
- ( P_i(t) ) — LTP of stock *i* at minute *t*  
- ( I(t) ) — official BankNifty index  
- ( w_i ) — learned stock weights  

### ⚙️ Ridge-Regression Formulation

We solve the ridge-regularized least-squares problem:

$$
\min_{w} \; \lVert y - Xw \rVert_2^2 + \lambda \lVert w \rVert_2^2
$$

Taking the derivative and setting to zero gives:

$$
(X^{\top} X + \lambda I) w = X^{\top} y
$$

Thus:

$$
w = (X^{\top} X + \lambda I)^{-1} X^{\top} y
$$

---

## 5) Notebooks — how they work
### 5.1 `making_dataset.ipynb`
1. **Glob** raw minute CSVs from a folder (e.g., `data/*.csv`).
2. **Filter** `exchange == 'NSECM'` and **group** by `name` (ticker).
3. Extract per‑ticker **LTP** (and optionally **volume**) series, align them by row index (minute order), and **concatenate side‑by‑side**.
4. **Export** the wide table to `finals_ltp.csv`.

### 5.2 `constructing_index.ipynb`
1. **Load** `finals_ltp.csv` and optional `weights.csv`.
2. If `weights.csv` is missing, **use fallback weights** embedded in the notebook (HDFCBANK, ICICIBANK, SBIN, etc.).
3. **Intersect** symbols between weights and data; **renormalize** the weights to sum to 1; **forward‑fill** tiny gaps.
4. Compute **RawCM = X·w**.
5. If `bn_official.csv` is provided, **align levels** at the **first overlapping bar**; else **Base‑100**.
6. **PCA fallback** kicks in automatically when no valid weights exist.
7. **Save** the constructed series (e.g., `cm_index_from_ltp.csv`).

---

## 6) Quickstart
### 6.1 Environment
```bash
python>=3.10
pip install pandas numpy matplotlib
```
(Only `numpy/pandas` are strictly required for the core path; `matplotlib` is used for quick plots.)

### 6.2 Prepare inputs
- Put all raw minute CSVs under a folder, e.g., `data/`.
- Ensure each raw file has at least: `date, time, exchange, name, ltp`.
- (Optional) Create `weights.csv` with **free‑float weights**:

```
symbol,weight
HDFCBANK,0.2710
ICICIBANK,0.2305
SBIN,0.1230
KOTAKBANK,0.1100
AXISBANK,0.0910
INDUSINDBK,0.0500
PNB,0.0310
BANKBARODA,0.0300
FEDERALBNK,0.0220
IDFCFIRSTB,0.0210
CANBK,0.0200
AUBANK,0.0015
```

### 6.3 Run
1. Open `making_dataset.ipynb` and run all cells → produces `finals_ltp.csv`.
2. Open `constructing_index.ipynb`:
   - Set `BN_OFFICIAL_PATH = "bn_official.csv"` if you want alignment.
   - Optionally place your `weights.csv` next to the notebook.
   - Run all cells → produces `cm_index_from_ltp.csv` (or `cm_index_mcap_weighted.csv`).

---

## 7) Outputs
- **Constructed index CSV** with columns:
  - `timestamp` (surrogate 1..N or real minute index if you wire it in),
  - `cm_index` (level‑aligned or Base‑100).
- (Optional) quick‑look plots inside the notebook.

---

## 8) Validation ideas (recommended)
- **Return correlation:** corr(∆log CM, ∆log Official).
- **Tracking error:** std( (∆CM − ∆Official) ).
- **Weight sanity:** sum to 1; major banks dominate weights; no symbol drift.
- **Gap checks:** count of NaNs before/after forward‑fill.

---


