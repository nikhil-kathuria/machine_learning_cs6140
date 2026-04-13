# Assignment 5 — Polluted Data, Missing Data, PCA, Ridge Logistic Regression

This assignment applies previously-built classifiers to harder variants of the spam and digits datasets (polluted features, missing values) and adds PCA-based dimensionality reduction. It also extends Assignment 2's logistic regression with L2 regularization (ridge).

## Files

### `Regression.py` — Extended logistic regression (with Ridge)
A copy of `Assignment_2/Regression.py` with two additions:

| Added function | Purpose |
|---|---|
| `ridgeLikelyhood(weights, data, labels)` | Log-likelihood minus L2 penalty: `LL - (cost / 2N) * ||w||²`. Used as the convergence metric for ridge training. |
| `ridgeGradient(data, labels)` | Training loop identical to `checkGradient` but calls `ridgeweights` and `ridgeLikelyhood`. Converges when `|j_old - j_new| < 0.0001`. |
| `ridgeweights(hypoth_V, weights, data, labels)` | Gradient step with L2 regularization: `w = w + λ*(Xᵀ(y-ŷ)) + (cost/N)*w`. `cost = 0.1`, `lam = 0.001`. |

> **Logistic regression cross-reference:** `checkGradient` (plain logistic) and `ridgeGradient` (ridge logistic) from this file are both called by `PollutedRegression.py`.

---

### `PollutedBayes.py` — Gaussian NB on polluted spam
Applies Gaussian Naive Bayes directly on the polluted feature dataset (`spam_polluted/`). Delegates entirely to `Assignment_3/NaiveBayes.computeU`, `predictGaussian`, and `accCalc` imported via `sys.path.insert`.

---

### `PollutedRegression.py` — Logistic regression on polluted spam
Runs both plain logistic (`runLogistic`) and ridge logistic (`runRidge`) from local `Regression.py` on the polluted spam dataset. Also contains a `runLibLinear` function that wraps `pyliblinear` (L1/L2 logistic via liblinear) — this requires the `pyliblinear` package and may not be functional.

---

### `PCA.py` — PCA dimensionality reduction + Gaussian NB
Uses `sklearn.decomposition.PCA` to reduce the polluted spam features to `10%` of original dimensions, then runs Gaussian NB (imported from `Assignment_3/NaiveBayes`). The reduction is fit on concatenated train+test, then split back — **note: this leaks test data into PCA fitting.**

---

### `missingNaive.py` — Bernoulli NB with missing values
Applies Bernoulli Naive Bayes (from `Assignment_3/NaiveBayes`) to the `missing/` dataset where some feature values are `NaN`. Computes column means skipping `NaN` entries (`computeMean`), then passes those means to `bernoulliDistribution` as the binarization threshold. The `predictBernoulli` function from Assignment 3 is used unchanged.

---

### `LineraDigits.py` — Linear logistic on MNIST digits (via pyliblinear)
Uses `pyliblinear` (liblinear wrapper) with an L2 logistic regression solver (`type=1`, i.e., Ridge) on Haar features of handwritten digits. Loads data from `Assignment_5/HF/` — the `s20train.txt` (20 training examples per class) and `htest.txt` (full test set).

---

### `ReadPolluted.py`, `readMissing.py`
Dataset-specific loaders:
- `ReadPolluted.readData(name)` — loads `train_feature.txt` / `test_feature.txt` (space-delimited, no labels column)
- `ReadPolluted.readLabels(name)` — loads `train_label.txt` / `test_label.txt`
- `readMissing.read(name)` — loads the missing-values dataset; `numpy.loadtxt` with `dtype=float` naturally produces `NaN` for missing entries

### `normalizedata.py`
Copy of Assignment 2 utility. Modifies matrix in place.

## Cross-assignment dependencies

| Source | Used here |
|---|---|
| `Assignment_3/NaiveBayes.py` | `PollutedBayes.py`, `PCA.py`, `missingNaive.py` — imported via `sys.path.insert` |
| `Assignment_2/Regression.py` (pattern) | `Regression.py` in this assignment extends it with ridge regression |
| `Assignment_5/HF/` Haar digit files | Reused by **Assignment 6** (`SVMDigits.py`, `LineraDigits.py`) and **Assignment 7** (`KNN_Distance.py`, `KNN_Window.py`, `KNN_DensityDigits.py`) |
