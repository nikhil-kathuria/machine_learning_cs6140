# Assignment 6 — SVM (SMO Implementation + libsvm + liblinear)

This assignment contains the **SVM implementations**: a from-scratch SMO solver in Java and two Python wrappers using external SVM libraries (libsvm and sklearn) for both spam and digits datasets.

## Where SVM lives

### Python — `SVMSpam.py`
Runs SVM on the spam dataset (10-fold CV). Two implementations:

**`performSVM(knl)`** — Uses **libsvm** (`svmutil` imported from `libsvm-3.20/python/`):
- Converts data to libsvm's `svm_problem` format
- Trains with `-t 2 -c 2 -g 2` (RBF kernel, C=2, gamma=2)
- Reports both test and train accuracy via `svm_predict`
- `knl` parameter is accepted but the model string is hardcoded (RBF always)

**`performSCI()`** — Uses **sklearn** `svm.SVC`:
- `SVC(kernel='rbf', C=10, tol=0.001)` or `LinearSVC()` (commented out)
- Simpler interface; normalization is commented out in this path

### Python — `SVMDigits.py`
Runs SVM on handwritten digits (Haar features from `Assignment_5/HF/`). Uses **libsvm** only:
- Loads train set (`s20train.txt` / `s20labels.txt`) and test set (`htest.txt` / `htestlabels.txt`)
- Trains with `-t 0` (linear kernel)
- Data is loaded as `dtype=int` — no normalization applied

### Python — `LineraDigits.py`
Uses **pyliblinear** (liblinear Python wrapper) with Ridge L2 logistic regression (`type=1`) on the same digits Haar data. Despite the name, this is logistic regression, not SVM. Same data loading pattern as `SVMDigits.py`.

---

### Java — SMO from scratch (`SMOSolver.java`, `SMOSimplified.java`, `SMOSpam.java`, `SMODigitsOne.java`, `SMODigitsRest.java`)
Full hand-written SMO (Sequential Minimal Optimization) SVM solver:

- **`SMOSolver.java`** — Core SMO implementation. Manages the `KernelCache` (stores `k11`, `k12`, `k22`) and implements the SMO dual update step. This is the central algorithm file.
- **`SMOSimplified.java`** — Simplified version of SMO for binary classification.
- **`SMOSpam.java`** — Runner: trains the SMO SVM on spam data.
- **`SMODigitsOne.java`** — One-vs-one SVM for digit classification.
- **`SMODigitsRest.java`** — One-vs-rest SVM for digit classification.
- **`DigitsLinear.java`** — Linear SVM variant for digits.

---

## Utility files

### `readData.py` / `normalizedata.py`
Copies of Assignment 2 utilities. Used by `SVMSpam.py`.

### `Normalize.java` / `ParseSpam.java`
Java-side data loading and normalization for the SMO experiments.

## Cross-assignment dependencies

| Dependency | Direction |
|---|---|
| `Assignment_1/spambase.data.txt` | Read by `SVMSpam.py` (hardcoded path) |
| `Assignment_5/HF/` digit files | Read by `SVMDigits.py` and `LineraDigits.py` |
| `Assignment_1` Java classes | All Java here imports `neu.ml.assignment1.*` |
| `libsvm-3.20/python/svmutil` | Required by `SVMSpam.py` and `SVMDigits.py` — must be present at the hardcoded path |
| `pyliblinear` | Required by `LineraDigits.py` |

## Result files
- `Digit_SVM_Linear.txt` / `Digits_SVM_rbf.txt` — Recorded accuracy results from the Java SMO runs
