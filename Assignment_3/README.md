# Assignment 3 — Naive Bayes, GDA, and EM Clustering

This assignment implements three generative classifiers and an EM algorithm for Gaussian mixture fitting.

## Files

### `GDA.py` — Gaussian Discriminant Analysis
Full GDA implementation using a shared covariance matrix. Key functions:

| Function | Purpose |
|---|---|
| `computeU(traindata, trainlabels)` | Computes per-class means U0 and U1 |
| `getPrior(labels)` | Computes class priors P(Y=0) and P(Y=1) |
| `pxGivenY(cov, det, inv, diff, nby2)` | Multivariate Gaussian density: `(2π)^(-n/2) * det^(-0.5) * exp(-0.5 * diffᵀ Σ⁻¹ diff)` |
| `getPrediction(...)` | MAP prediction: compares `P(Y=0)*P(X\|Y=0)` vs `P(Y=1)*P(X\|Y=1)` |
| `runGDA(name)` | 10-fold CV on spam; computes covariance with `numpy.cov`, inverts with `numpy.linalg.pinv` |
| `accCalc(predictions, labels)` | Simple accuracy — **imported by NaiveBayes, Assignment 5 (PollutedBayes, PCA), and Assignment 5 (missingNaive)** |

> **Cross-assignment dependency:** `computeU`, `getPrior`, `accCalc` from `GDA.py` are imported by `NaiveBayes.py` in this assignment, and `computeU`, `predictGaussian`, `accCalc` from `NaiveBayes.py` are imported by **Assignment 5** (`PollutedBayes.py`, `PCA.py`, `missingNaive.py`) via `sys.path.insert`.

---

### `NaiveBayes.py` — Naive Bayes (three variants)
Implements three different feature likelihood models — all share the same 10-fold CV loop structure:

**Gaussian NB** (`runGau`): Uses per-feature variance (diagonal covariance). Delegates to `computeU` from `GDA.py` for means. Calls `predictGaussian` which sums log `P(x_i | Y=k)` independently per feature. Handles zero-variance features with `fixvar`.

**Bernoulli NB** (`runBer`): Binarizes each feature relative to its mean (above/below mean). `bernoulliDistribution` computes `P(x_i=0|Y)` and `P(x_i=1|Y)` with +1 Laplace smoothing.

**Histogram Binning NB** (`runBinning`): Uses `meanbins.bindata` to create non-uniform bins based on class means. `predictBinning` applies Laplace-smoothed bin counts.

---

### `em.py` — EM for 2 Gaussians
Fits a 2-component Gaussian Mixture Model via EM. Operates on `2gaussian.txt` (4000 points).

| Function | Purpose |
|---|---|
| `init2M(data, p1)` | Initializes model parameters by splitting data at row `p1` — each half gets its own mean, covariance, and prior |
| `computeE(data, M1, M2)` | E-step: computes soft assignments `z1`, `z2` using Bayes' rule on Gaussian densities |
| `computeParam(data, Z, U)` | M-step for one component: updates mean, covariance, and prior from soft assignments |
| `computeM(data, Zim, U1, U2)` | M-step: calls `computeParam` for both components |

Runs 30 EM iterations and prints the final model parameters.

### `em2.py`
Identical to `em.py` — same code, same data. Appears to be a copy made during development.

### `em3.py` — EM for 3 Gaussians
Extension of `em.py` to a 3-component GMM for `3gaussian.txt` (7500 points). `init2M` takes two split points `(p1, p2)`. The E-step denominator sums all three component likelihoods. Runs 110 iterations.

---

### `binning.py` — Uniform histogram binning
Computes equal-width bins between `[min, max]` for each feature column. Returns `(spamlist, hamlist, boundary)` per feature. Used by `NaiveBayes.predictBinning`.

### `meanbins.py` — Mean-based binning
Non-uniform binning strategy: boundaries are set at `[min, mean(class1), mean(all), mean(class0), max]` — places bin boundaries at statistically meaningful values. Used by `NaiveBayes.runBinning`.

### `equalbins.py`
Similar to `binning.py` — alternate uniform-bin implementation. Not imported by other files.

### `plotROC.py`
Copy of `Assignment_2/plotROC.py`. Imported by `NaiveBayes.py`.

### `readData.py` / `normalizedata.py`
Copies of Assignment 2 utilities. `readData.py` adds `readHaar` for loading Haar feature files.

## Cross-assignment dependencies

| Dependency | Used by |
|---|---|
| `NaiveBayes.computeU`, `predictGaussian`, `accCalc` | Assignment 5: `PollutedBayes.py`, `PCA.py` |
| `NaiveBayes.partionMatrix`, `bernoulliDistribution`, `predictBernoulli`, `accCalc` | Assignment 5: `missingNaive.py` |
| `GDA.computeU`, `getPrior`, `accCalc` | Imported within this assignment by `NaiveBayes.py` |

Assignment 5 uses `sys.path.insert(0, '.../Assignment_3')` to import from this directory directly.
