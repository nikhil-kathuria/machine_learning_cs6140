# Assignment 2 — Logistic/Linear Regression, Perceptron, Neural Network

This assignment contains the **primary implementations of logistic regression and linear regression** for the whole course. The `Regression.py` file is reused (copied) in Assignment 5.

## Where logistic regression lives

**`Regression.py`** — Core implementation. Key functions:

| Function | Purpose |
|---|---|
| `sigmoid(value)` | Scalar sigmoid: `1 / (1 + exp(-x))` |
| `logistichypoth(weights, data)` | Applies sigmoid element-wise after a linear projection |
| `linearhypoth(weights, data)` | Linear hypothesis: `data · weights` |
| `checkGradient(data, labels, linear)` | **Gradient ascent training loop** — runs until `|j_old - j_new| < 0.0001`. Pass `linear=True` for MSE (linear regression), `linear=False` for log-likelihood (logistic regression). This is the core training function for both models. |
| `updateWeights(hypoth_V, weights, data, labels)` | Single gradient step: `w = w + λ * Xᵀ(y - ŷ)`, where `λ = 0.0001` |
| `likelyhood(weights, data, labels)` | Log-likelihood convergence metric for logistic |
| `computeCost(weights, data, labels)` | MSE convergence metric for linear |
| `predict(weights, test)` | Adds intercept column, returns raw scores |
| `accCalc(prediction, labels, threshold)` | Returns accuracy and prints TP/FP/TN/FN |
| `runLogisticRegressin()` | Runs 10-fold CV logistic regression on spam |
| `runLinearRegression()` | Runs linear regression on housing dataset |

Learning rate (`lam`) is a module-level constant set separately for linear (`0.0001`) and logistic (`0.0005`) — you need to change the constant between runs.

## Other files

### `perceptron.py`
Primal perceptron on a small 2D dataset (`perceptronData.txt`). Uses the mistake-bound update rule: for each misclassified point, `w = w + λ * x_i`. Labels must be ±1. The `missandData` function folds labels into the data matrix (multiplies rows with label -1 by -1) so all rows should satisfy `w · x > 0`.

### `NeuralNetwork.py` / `NN.py` / `NN_Scalar.py`
Three versions of the same autoencoder experiment (8-input → 3-hidden → 8-output), all implementing backpropagation manually:
- **`NN.py`** — Clean vectorized version using `numpy` matrix ops. This is the one that saves `Hidden.txt` and `Output.txt`.
- **`NeuralNetwork.py`** — Earlier version; imports `sigmoid` from `Regression.py`. Saves biases `B1`/`B2` to separate files. Uses `LR = 0.2`.
- **`NN_Scalar.py`** — Fully scalar nested-loop version (explicit `for` loops over every weight). Has a syntax error in `getFinal` (`out[rowk][]`) — this file does not run.

All three use the 8×8 identity matrix as training data (the autoencoder learns to reconstruct the identity through a 3-node bottleneck). Backprop delta rule: `E3 = L3*(1-L3)*(target - L3)`, `E2 = L2*(1-L2)*(Wjk · E3)`.

### `plotROC.py`
ROC curve and AUC calculation from scratch. `rocAndAuc` zips labels with predictions, sorts by score, then iterates thresholds to compute TPR/FPR at each point. AUC computed with `numpy.trapz`.

### `readData.py`
- `readSpam(name)` — Loads full spam CSV, shuffles row indices, returns a `(bucketmap, full_matrix)` pair where `bucketmap` is a 10-fold partition dictionary.
- `extractMatrix(bucketmap, keys, full)` — Assembles train or test matrix from selected bucket keys.
- `readHouse`, `readSmallSpam`, `readPerceptron` — Simple loaders returning `(data, labels)`.

### `normalizedata.py`
Min-max normalization per column. Columns with `min == max` are zeroed out. **Modifies the matrix in place.**

### `tmp.py`
Debug script — loads saved `Wjk` and `Hidden` weight files and recomputes the NN output to verify correctness.

## Cross-assignment dependencies

- `Regression.py` is copied (with modifications) into **Assignment 5** as `Regression.py`, adding `ridgeGradient` and `ridgeLikelyhood` for L2-regularized logistic regression.
- `readData.py` and `normalizedata.py` are copied into every subsequent assignment (3–7). The spam 10-fold CV split pattern from `readSpam` + `extractMatrix` is used universally.
- `plotROC.py` is copied into **Assignment 3**.
- `NeuralNetwork.py` imports `sigmoid` / `sigmoidAll` directly from `Regression.py` — it will not run if `Regression.py` is not in the same directory.
