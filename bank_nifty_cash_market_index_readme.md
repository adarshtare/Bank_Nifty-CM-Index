# BankNifty Cash‑Market Index — Ridge Regression Weighting (Tracking‑Error Minimization)

Rebuild a **BankNifty‑like cash‑market index** from constituent bank stock prices (LTPs), using a **ridge‑regularized least‑squares approach** to estimate stock weights that best replicate the official Bank Nifty index movement.

---
## 📘 Methodology

We estimate the weights **w** that make our constructed index best replicate the official BankNifty.

The optimization problem is:

$$
\min_{w \ge 0,\; \sum_i w_i = 1}
\sum_{t \in \mathcal{T}}
\left( I(t) - \sum_i w_i\, P_i(t) \right)^2
$$

Where:  
- \( P_i(t) \) — LTP of stock *i* at minute *t*  
- \( I(t) \) — official BankNifty index  
- \( w_i \) — learned stock weights  

After solving for weights, the custom index is:

$$
\text{cm\_index}(t) = \alpha \sum_i w_i\, P_i(t)
$$

Here \( \alpha \) is a level alignment factor:

$$
\alpha = \frac{I(t_0)}{\sum_i w_i\, P_i(t_0)}
$$

---

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

### 🧠 Implementation (Python)

```python
import numpy as np

# X: matrix of LTPs (T x N)
# y: official BankNifty series (T,)
# lambda_: ridge regularization parameter

XtX = X.T @ X + lambda_ * np.eye(X.shape[1])
Xty = X.T @ y
w = np.linalg.solve(XtX, Xty)

# Construct synthetic index
cm_index = X @ w
cm_index *= y[0] / cm_index[0]  # level alignment

## 1) Overview
This project reconstructs the **BankNifty Index** using per‑minute LTP data from its 12 constituent banks. Instead of fixed market‑cap weights or PCA loadings, the model learns **data‑driven weights** $$ \( w_i \) $$ by minimizing the **tracking error** between the official BankNifty and a weighted combination of constituent stock prices.

---

## 2) Objective Formulation
We seek non‑negative weights $$ \( w_i \) $$(summing to 1) that make the constructed index track the official index $$ \( I(t) \) $$ as closely as possible over a calibration window $$ \( \mathcal{T} \)  :

$$
\[
\min_{w \ge 0,\; \sum_i w_i = 1} \sum_{t \in \mathcal{T}} \Big( I(t) - \sum_i w_i P_i(t) \Big)^2
\]
$$

where:
$$
 \( P_i(t) \):
$$ 
- LTP of stock *i* at minute *t* $$  
$$
\( I(t) \):
$$
-  official BankNifty index value at minute *t* $$

The constructed index is then given by:

\[
\text{cm\_index}(t) = \alpha \sum_i w_i P_i(t)
\]

Here \( \alpha \) is a level‑alignment factor ensuring your synthetic index sits on the same numerical scale as the official one (≈ divisor adjustment).

---

## 3) Ridge‑Regularized Solution
The optimization problem can be solved in matrix form using **ridge regression** (L2‑regularized least squares):

Let:
- \( X \): matrix of constituent prices of shape (T_cal × N)
- \( y \): vector of official BankNifty index values of length T_cal

We want to minimize:
\[
\min_w \; \| y - Xw \|_2^2 + \lambda \|w\|_2^2
\]

Taking the derivative and setting to zero gives the **normal equations**:
\[
(X^T X + \lambda I) w = X^T y
\]

Hence the solution is:
\[
\boxed{w = (X^T X + \lambda I)^{-1} X^T y}
\]

This produces the **ridge‑regularized least‑squares estimate of weights**, avoiding overfitting and instability in case of collinear stocks or noisy minute data.

In code:
```python
XtX = X.T @ X + lambda_ * np.eye(X.shape[1])
Xty = X.T @ y
w = np.linalg.solve(XtX, Xty)
```

Interpretation:
- `X.T @ y` measures **how strongly each stock covaries** with the official index.
- The ridge term \( \lambda I \) stabilizes inversion and shrinks extreme weights.
- `np.linalg.solve()` efficiently computes the weight vector.

---

## 4) Why Ridge Regularization?
Without regularization, weights can oscillate wildly due to multicollinearity (stocks move together). Ridge regression:
- Adds stability by penalizing large weights.
- Controls variance–bias trade‑off.
- Produces smoother and more interpretable tracking weights.

Tuning \( \lambda \):  
- \( \lambda = 0 \) → pure OLS fit.  
- Larger \( \lambda \) → smoother weights, slightly less precise tracking.

---

## 5) Workflow Summary
### a) `making_dataset.ipynb`
- Loads multiple NSE minute‑data files.
- Filters `exchange == 'NSECM'`.
- Groups by ticker, extracts LTPs.
- Concatenates all tickers side‑by‑side → `finals_ltp.csv`.

### b) `constructing_index.ipynb`
- Loads `finals_ltp.csv` and official index `bn_official.csv`.
- Normalizes and aligns both on common timestamps.
- Solves ridge regression for \( w \).
- Computes `cm_index(t) = α Σ_i w_i P_i(t)`.
- Plots and exports `cm_index_from_ltp.csv`.

---

## 6) Data Inputs
### `finals_ltp.csv`
| Column | Description |
|---------|--------------|
| AUBANK | LTP of AU Small Finance Bank |
| AXISBANK | LTP of Axis Bank |
| … | … |
| SBIN | LTP of State Bank of India |

### `bn_official.csv`
| Column | Description |
|---------|--------------|
| ltp | Official BankNifty index value |
| date, time | Timestamp fields used for alignment |

---

## 7) Outputs
### `cm_index_from_ltp.csv`
| Column | Description |
|---------|--------------|
| timestamp | Minute or sequence index |
| cm_index | Reconstructed index level aligned with official series |

Also produces diagnostic plots:
- Tracking comparison (Official vs Constructed)
- Residual (tracking error)
- Learned weights chart

---

## 8) Evaluation Metrics
- **Return correlation:** corr(Δlog CM, Δlog Official)
- **Tracking error:** std(ΔCM − ΔOfficial)
- **Weight distribution:** check non‑negative, sum to 1.

---

## 9) Interpretation
This ridge‑based approach builds a **statistical replica** of the official BankNifty, learning from the data rather than following fixed market‑cap weights. It is useful for:
- ETF or derivative replication studies
- Custom sector index modeling
- Factor or tracking‑error analysis

It approximates the **shape** of the real index extremely well, though not an official definition (since free‑float market caps are proprietary).

---

## 10) Quickstart
```bash
pip install pandas numpy matplotlib
python3
```

```python
# Load CSVs
X = pd.read_csv('finals_ltp.csv').values
y = pd.read_csv('bn_official.csv')['ltp'].values[:len(X)]

# Center (optional)
Xc = X - X.mean(axis=0)
yc = y - y.mean()

# Ridge regression weights
lambda_ = 1e-3
w = np.linalg.solve(Xc.T @ Xc + lambda_ * np.eye(X.shape[1]), Xc.T @ yc)

# Construct index
cm_index = X @ w
cm_index *= y[0] / cm_index[0]   # Level alignment
```

---

## 11) Notes
- If official data (`bn_official.csv`) is missing, model cannot calibrate weights.
- Regularization parameter \( \lambda \) controls smoothness.
- Scaling by \( \alpha \) ensures the same base level as official series.

---

## 12) Summary
| Concept | Description |
|----------|--------------|
| **Goal** | Reconstruct BankNifty using ridge‑regularized least‑squares weights |
| **Input** | Per‑minute LTPs of 12 constituent banks |
| **Output** | Synthetic CM index tracking official BankNifty |
| **Key math** | \( w = (X^T X + λI)^{-1} X^T y \) |
| **Advantage** | Stable, data‑driven, minimal tracking error |

---

**Author:** [Your Name]  
**License:** MIT / Academic Use  
**Keywords:** BankNifty, Ridge Regression, Tracking Error, Index Replication, Quant Finance

