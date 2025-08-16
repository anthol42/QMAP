*src.qmap.benchmark*
# Class: `QMAPBenchmark`

**Description:** Class representing the QMAP benchmark testing dataset. It is a subclass of `torch.utils.data.Dataset`, so it can be
easily used with PyTorch's DataLoader. However, it is easy to extract the sequences and the labels from it to use
it with other libraries such as tensorflow or keras. You can do this by using the `inputs` and `targets` attributes.

Once you have the predictions of your model, you can call the `compute_metrics` method to compute the metrics of
your model on the given test set. You can also evaluate the performances of your model on subsets like
`high_complexity`, `low_complexity` or `high_efficiency`. You can select the subset from the attributes of the
same name. The subset have the same interface as the `QMAPBenchmark` class, so you can use them with the same
methods!

To use the benchmark, you must select at least the split (from 0 to 4) and the threshold (55 or 60). It is highly
recommended to test your model on all splits to get a better estimate of its real-world performance and to
accurately compare it with other models. To do so, you must use the same hyperparameters, but change the training
and validation dataset. For each split, use the `get_train_mask` method to get a mask indicating which sequences
can be used in the training set and validation set. This mask will be True for sequences that are allowed in the
training set and False for sequences that are too similar to a sequence in the test set. Train your model on the
subset of your training dataset where the mask is True and evaluate it on the benchmark dataset. Do this for all
splits. See the example section for more details.

Thresholds:

- 55: This threshold enables a split that is considered natural as it have a maximum identity distribution between
the train and test set similar to natural independent peptide datasets.

- 60: This threshold enables a harder split because it increases the diversity of the test set. Even if the
maximum identity distribution is shifted to more similar sequences between train and test compared to the natural
split (55), it is considered conservative as models do not perform as well on this split. It is recommended to use
this split as it gives a more conservative estimate of the model's real-world performance.

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
## Method: `get_train_mask()`

```python
get_train_mask(self, sequences: list[str], encoder_batch_size: int = 512, align_batch_size: int = 0, force_cpu: bool = False) -> ndarray:
```

**Description:** Returns a mask indicating which sequences can be in the training set because they are not too similar to any
other sequence in the test set. It returns a boolean mask where True means that the sequence is allowed in the
training / validation set and False means that the sequence is too similar to a sequence in the test set and
must be excluded.

**Parameters:**
- `sequences`: The sequences to check.
- `encoder_batch_size`: The batch size to use for encoding. Change this value if you run out of memory.
- `align_batch_size`: The batch size to use for alignment. If 0, the batch size will be the full dataset. Change this value if you run out of memory.
- `force_cpu`: If True, the alignment will be forced to run on CPU.

**Return:**
- True if the sequence is allowed and False otherwise.
