# Assignment 4 — Boosting, Bagging, ECOC, Active Learning (Java + Python)

This assignment is primarily Java. The single Python file is a results visualizer.

## Python files

### `PlotStats.py`
Parses `AdaBoostingResults.txt` and plots three separate charts using matplotlib:
- Round error per boosting iteration
- Train vs. test error per iteration
- AUC per iteration

No ML logic — purely a visualization utility for the Java output.

## Java files (overview)

The Java code imports `neu.ml.assignment1.*` (`CrossTrainData`, `DecisionTree`, `NormalizeData`, `ParserAndBuildMatrix`) for all data handling.

- **`AdaBoosting.java`** — AdaBoost with `DecisionStump` as the weak learner. Writes per-round stats (round error, train error, test error, AUC) to `AdaBoostingResults.txt`, which `PlotStats.py` reads.
- **`DecisionStump.java`** — Weighted one-level decision tree (stump) used by AdaBoost. Supports a random-feature mode (for Bagging).
- **`Bagging.java`** — Bootstrap aggregation using random decision stumps.
- **`BoostedTrees.java`** — AdaBoost variant using full decision trees as weak learners instead of stumps.
- **`HybridStumps.java`** — Variant that mixes stump types.
- **`BoostEcoc.java`** / **`EcocBoosting.java`** / **`ParseEcoc.java`** — Error-Correcting Output Codes for multiclass boosting. `ECOC/` directory contains the ECOC code matrices.
- **`ActiveLearning.java`** — Pool-based active learning: selects the most uncertain samples for labeling.
- **`AUC.java`** — AUC computation helper.
- **`IterationStats.java`** — Data class for per-round metrics.
- **`ParseSpam.java`** — Spam-specific data parser.
- **`WriteToFile.java`** — Writes result arrays to text files.
- **`UCIBoosting.java`** / **`Run8NewsGroup.java`** / **`RunMINST.java`** / **`RunAdaPolluted.java`** — Runner classes for different datasets.

## Cross-assignment dependencies

- All Java in this assignment depends on `neu.ml.assignment1.*` from Assignment 1.
- `AdaBoostingResults.txt` is the bridge between Java (`AdaBoosting.java`) and Python (`PlotStats.py`).
