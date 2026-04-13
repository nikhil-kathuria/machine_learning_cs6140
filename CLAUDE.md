# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Northeastern University CS6140 Machine Learning course repository containing seven assignments, each implementing ML algorithms from scratch. Each assignment is a self-contained directory with both Java and Python implementations.

## Running Code

### Python
Each Python script is run directly. Scripts import shared utilities (`readData.py`, `normalizedata.py`) from the same assignment directory.

```bash
python Assignment_7/KNN_5Spam.py
python Assignment_3/em.py
```

### Java
Java files use the `neu.ml.assignmentN` package structure and depend on the [EJML library](http://ejml.org/) (`SimpleMatrix`). Compile and run from the assignment directory:

```bash
javac -cp .:ejml.jar Assignment_1/ExecuteClassification.java
java -cp .:ejml.jar neu.ml.assignment1.ExecuteClassification
```

Later Java assignments (4+) import classes from `neu.ml.assignment1` (e.g., `DecisionTree`, `CrossTrainData`, `NormalizeData`), so those must be on the classpath.

## Codebase Architecture

### Shared Utilities (per-assignment)
Each assignment has local copies of:
- `readData.py` — loads datasets (`readSpam`, `readHouse`, `readHaar`, `readSpam` returns 10-fold CV buckets)
- `normalizedata.py` — min-max feature normalization

### Data Files
Primary datasets used across assignments:
- **Spam** (`spamtrain.txt` / `spamtest.txt`, comma-delimited, last column = label) — binary classification
- **Housing** (`housing_train.txt` / `housing_test.txt`) — regression
- **MNIST/Digits** (`htrain.txt` / `htestlabels.txt`, Haar features) — digit classification

### Assignment Progression

| Assignment | Topics | Key files |
|---|---|---|
| 1 | Decision Trees, Linear/Logistic Regression | `DecisionTree.java`, `Regression.py` |
| 2 | Neural Networks, Perceptron | `NeuralNetwork.py`, `perceptron.py` |
| 3 | Naive Bayes, GDA, EM clustering | `NaiveBayes.py`, `GDA.py`, `em.py` |
| 4 | AdaBoost, Bagging, ECOC, Active Learning | `AdaBoosting.java`, `DecisionStump.java`, `Bagging.java` |
| 5 | PCA, Polluted data, Missing data | `PCA.py`, `missingNaive.py`, `PollutedRegression.py` |
| 6 | SVM (SMO implementation + libsvm) | `SMOSolver.java`, `SVMSpam.py`, `SVMDigits.py` |
| 7 | KNN (distance/density/window), Dual Perceptron, Relief | `KNN_5Spam.py`, `DualPerceptron.py`, `Relief.py` |

### Java Cross-Assignment Dependencies
Assignment 4+ Java code imports from `neu.ml.assignment1`:
- `CrossTrainData` — holds train/test split with bucket maps
- `DecisionTree` / `DecisionStump` — tree learners
- `NormalizeData`, `ParserAndBuildMatrix`

### Hardcoded Paths
Many files contain hardcoded absolute paths (e.g., `/Users/nikhilk/Documents/NEU_MSCS/ML/...`). Update these to match the local environment before running. The `sys.path.insert` calls in Python files may also need updating when importing across assignment directories.
