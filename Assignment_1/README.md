# Assignment 1 — Decision Trees & Data Infrastructure (Java only)

This assignment is **Java only** — there are no Python files. It establishes the core data structures and the Decision Tree classifier that later Java assignments (4, 5, 6) build on top of.

## Files

### Core algorithm
- **`DecisionTree.java`** — The main decision tree implementation. Builds classification and regression trees using information gain / variance reduction.
- **`ExecuteClassification.java`** — Runs the Decision Tree on the spam dataset (10-fold CV using `CrossTrainData`).
- **`ExecuteRegression.java`** — Runs the Decision Tree on the housing dataset for regression.
- **`PredictClassification.java`** / **`PredictRegression.java`** — Prediction logic separated from tree construction.

### Shared infrastructure (imported by later assignments)
These classes are imported as `neu.ml.assignment1.*` in Assignments 4, 5, and 6:
- **`CrossTrainData.java`** — Holds the full dataset and a `bucketmap` (10-fold CV split). This is the primary data container passed around in all later Java assignments.
- **`ParserAndBuildMatrix.java`** — Reads raw data files into a matrix backed by `SimpleMatrix` (EJML).
- **`NormalizeData.java`** — Min-max normalization, mirrors what `normalizedata.py` does in Python.

### Data files
- `spambase.data.txt` — Full spam dataset (comma-delimited, last column = label 0/1)
- `spamtrain.txt` / `spamtest.txt` — Smaller pre-split spam sets
- `housing_train.txt` / `housing_test.txt` — Boston housing regression dataset

> **Cross-assignment dependency:** `CrossTrainData`, `DecisionTree`, `NormalizeData`, and `ParserAndBuildMatrix` from this assignment are reused by the Java code in Assignments 4, 5, and 6. The `spambase.data.txt` and housing data files are also directly referenced (via hardcoded paths) by Python scripts in Assignments 2–7.
