*src.qmap.benchmark*
# Class: `BenchmarkSubset`

**Description:** Base class of the QMAP benchmark class. It provides a common interface for the benchmark dataset and the subsets.

## Method: `accuracy()`

```python
accuracy(self, predictions: ndarray) -> float:
```

**Description:** Compute the accuracy of the predictions. A good prediction is one that is within the MIC range if a range is
provided. This method only work with MIC datasets (It does not work with Hemolytic or Cytotoxic datasets).

**Parameters:**
- `predictions`: The predictions to evaluate. It should have the same length and order as this dataset.

**Return:**
- The accuracy of the predictions.
## Method: `compute_metrics()`

```python
compute_metrics(self, predictions: ndarray, log: bool = True) -> typing.Union[QMAPRegressionMetrics, QMAPClassificationMetrics]:
```

**Description:** Compute the QMAP metrics given the predictions of the model. If the dataset type is MIC, it will return the
following metrics:
- RMSE
- MSE
- MAE
- R2
- Spearman correlation
- Kendall's tau
- Pearson correlation

If the dataset type is Hemolytic or Cytotoxic, it will return the following metrics:
- Balanced accuracy
- Precision
- Recall
- F1 score
- Matthews correlation coefficient [MCC]

**Note**:

This does not include the accuracy metric, which is computed separately.

**Parameters:**
- `predictions`: The predictions to evaluate. It should have the same length and order as this dataset.
- `log`: If true, apply a log10 on the targets.

**Return:**
- A QMAPMetrics object containing all the metrics.
