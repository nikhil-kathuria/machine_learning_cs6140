# Assignment 7 — KNN, Dual Perceptron, Relief Feature Selection

This assignment implements non-parametric methods: three variants of KNN (distance-based, window-based, density-based), dual perceptron with kernel, and the Relief feature-selection algorithm.

## Files

### `KNN_5Spam.py` — K-Nearest Neighbors (distance, fixed K)
Binary spam classifier using 10-fold CV. Evaluates three K values simultaneously (`[54, 55, 37, 47, 19]` — a feature list used elsewhere; the actual K values tried are `[1, 3, 7]` style). 

Key logic in `computeKernelSpam`:
- Builds a `smap` dict mapping `{row_index: (distance, label)}` for every training point
- Sorts by distance, takes the top-K, majority votes

Also includes a `computeKernelDigits` variant that uses a **polynomial kernel** `(γ * xᵀy + c)^d` as the distance metric for the digit dataset (higher polynomial score = closer neighbor, so sort descending).

Uses the hand-rolled `euclidianDistance` (manual loop, not `scipy.euclidean`).

---

### `KNN_Distance.py` — KNN with pluggable distance/kernel
More complete version of KNN for both spam and digits. Supports multiple distance metrics:

| Function | Metric |
|---|---|
| `euclidianDistance` | Standard L2 (uses `scipy.euclidean`) |
| `gaussian(testv, trainv)` | `exp(-d² / σ²)` with `σ=1` — similarity, not distance |
| `ploynomial(testv, trainv)` | `(γ * xᵀy + c)^d` with `γ=0.1`, `c=0.25`, `d=2` |

`predictSpam` / `predictDigits` sort and majority-vote top K. `predictDigits` takes `mbool` to control sort direction (descending for similarity-based kernels, ascending for distance-based).

Comments in the file record which settings work: "Do not Normalize for Spam and use your own euclidean", "polynomial gamma=.1, coefficient=.25, K=7 → 58% accuracy".

---

### `KNN_Window.py` — KNN with window (radius-based)
Instead of fixing K, includes all training points within a distance threshold. `computeKernelSpam` keeps only points where `distance <= num`. `computeKernelDigits` uses cosine distance with a threshold of `0.83`. The `predictSpam` / `predictDigits` functions here vote over all points inside the window (no K parameter).

---

### `KNN_DensitySpam.py` — Kernel density estimation (spam, binary)
Not strictly KNN — implements a non-parametric Bayes classifier using Gaussian kernel density:
- Splits training data into class-0 and class-1 matrices
- For each test point, estimates `p(x | Y=0)` and `p(x | Y=1)` by summing `gaussian(test, train_i)` over all class members and normalizing by class size
- Prediction: `argmax_c P(Y=c) * p(x | Y=c)`

---

### `KNN_DensityDigits.py` — Kernel density estimation (digits, multiclass)
Same density estimation approach as `KNN_DensitySpam.py` but extended to 10 digit classes. `partionMatrix` groups training data by digit label into a dict. `computeKernelDigits` estimates per-class density and multiplies by class prior.

Tested with both Gaussian (`σ=1`) and polynomial (`γ=0.5`) kernels — comments note "Gaussian → do not normalize, ~accuracy achievable".

---

### `DualPerceptron.py` — Dual-form perceptron with Gaussian kernel
Implements the dual representation of the perceptron, where the decision boundary is expressed as a weighted sum of kernelized training points rather than explicit weights.

Core algorithm in `dualPerceptron`:
- Maintains a mistake count vector `mvec` and a bias `B`
- For each point, computes `misVal = (Σ_i mvec[i] * y_i * K(x_i, x)) + B) * y`
- If `misVal <= 0` (misclassified): increment `mvec[row]` by 1 and update `B += y_row`

Kernel used: `gaussian(sigma=0.1)` — a very tight kernel that memorizes the training set. Tested on both `perceptronData.txt` (linearly separable) and `spiral.txt` (non-linearly separable).

`performupdate` is an older non-working variant (missing the `B` parameter).

---

### `Relief.py` — Relief feature selection
Ranks features by their ability to discriminate between classes. Core in `reliefAlgo`:

For a random sample of 1000 training points, finds the nearest same-class neighbor (near hit) and nearest different-class neighbor (near miss), then updates feature weights:
- If label=0: `weight -= (x - nearHit)² + (x - nearMiss)²` (sign pattern from binary Relief)
- Corrected Relief formula: same-class differences should decrease weight, cross-class should increase

Sorts features by final weight and prints the 10 lowest-ranked (least useful) features. Results saved to `reliefeatures.txt`.

`partionMatrix` is a local copy of the same utility in Assignment 3's `NaiveBayes.py`.

---

### `readData.py` / `normalizedata.py`
Copies of Assignment 2 utilities. `readData.py` here includes `readHaar` and `readSmallSpam`.

## Cross-assignment dependencies

| Data source | Used by |
|---|---|
| `Assignment_1/spambase.data.txt` | `KNN_5Spam.py`, `KNN_Distance.py`, `KNN_DensitySpam.py`, `KNN_Window.py`, `Relief.py` |
| `Assignment_5/HF/` Haar digit files | `KNN_Distance.py`, `KNN_Window.py`, `KNN_DensityDigits.py` |
| `Assignment_7/perceptronData.txt` | `DualPerceptron.py` (same 2D linearly separable set used in Assignment 2) |
| `Assignment_7/spiral.txt` | `DualPerceptron.py` — non-linearly separable test case requiring the kernel |
