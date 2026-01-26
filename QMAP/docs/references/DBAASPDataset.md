*src.qmap.benchmark*
# Class: `DBAASPDataset`

**Description:** Class representing a the DBAASP dataset used to build the benchmark.

## Example:
dataset = DBAASPDataset(...)

dataset_filtered = (dataset
           .with_bacterial_targets(["Escherichia coli", "Staphylococcus aureus", "Pseudomonas aeruginosa"])
           .with_efficiency_below(10.0)
           .with_common_only()
           .with_l_aa_only()
           .with_terminal_modification(None, None)
           )
print(dataset)

# You can also extend the dataset with your own samples:
dataset_extended = dataset_filtered + DBAASPDataset(your_own_data)
# Or with the extend method
dataset_extended = dataset_filtered.extend(DBAASPDataset(your_own_data))

# You can also index the dataset like a list, or a numpy array:
first_sample = dataset[0]
some_samples = dataset[1:10]
boolean_indexed_samples = dataset[np.array([True, False, True, False, ...])]
only_certain_samples = dataset[[0, 2, 5, 7]]

## Method: `compute_metrics()`

```python
compute_metrics(self, predictions: list[dict[str, float]], log: bool = True, mean_metrics: bool = True) -> dict[str, QMAPRegressionMetrics]:
```

**Description:** Compute metrics on the dataset given the predictions. The predictions must be in a specific format:

A list of dictionaries, where each dictionary corresponds to the prediction for a sample, and each key in the
dictionary is associated to a property name. These keys can be any bacterial target name, or hc50 for
hemolytic activity. If a label exist for the sample and the property, it will be used to compute the metrics.
Otherwise, the prediction is ignored. Please, make sure that all the keys do exist, otherwise the function will
fail silently by returning nan metrics for this property.

> **IMPORTANT:**
> The order of the predictions must match the order of the samples in the dataset.
> Thus, the length of the predictions list must be equal to the length of the dataset.



are made in the log form. This is recommended.

properties. The key is named 'mean'.

**Parameters:**
- `predictions`: The predictions to evaluate.
- `log`: If True, apply a log10 on the targets before computing the metrics. This means that the prediction
- `mean_metrics`: If True, a key will be added to the output containing the mean metrics across all

**Return:**
- A dictionary of QMAPMetrics objects, one for each property predicted. The key is the property name,
and the value the set of metrics.
## Method: `extend()`

```python
extend(self, other: DBAASPDataset) -> DBAASPDataset:
```

**Description:** Extend the dataset with another DBAASPDataset.

## Method: `filter()`

```python
filter(self, mapper: Callable[Sample'>], bool]) -> DBAASPDataset:
```

**Description:** Filter the dataset using a mapper function that takes a Sample and returns a boolean.
A sample is kept if the mapper returns True, and drop otherwise.

## Method: `get_train_mask()`

```python
get_train_mask(self, sequences: list[str], threshold: float = 0.6, matrix: str = "blosum45", gap_open: int = 5, gap_extension: int = 1, use_cache: bool = True, show_progress: bool = True, num_threads: typing.Union[int, NoneType] = None) -> ndarray:
```

**Description:** Returns a mask indicating which sequences can be in the training set because they are not too similar to any
other sequence in the test set. It returns a boolean mask where True means that the sequence is allowed in the
training / validation set and False means that the sequence is too similar to a sequence in the test set and
must be excluded.



Supported: blosum{30, 35, 40, 45, 50, 55, 60, 62, 65, 70, 75, 80, 85, 90, 95, 100}
Also: pam{10-500} in steps of 10

**Parameters:**
- `sequences`: The training sequences to check against the benchmark test set.
- `threshold`: Minimum similarity threshold to save the edge.
- `matrix`: Substitution matrix name (default: "blosum45")
- `gap_open`: Gap opening penalty
- `gap_extension`: Gap extension penalty
- `use_cache`: Whether to use caching (default: True)
- `show_progress`: Whether to show progress bar
- `num_threads`: Number of threads to use for parallel computation (default: None = all available cores)

**Return:**
- A binary mask where True means that the sequence is allowed in the training set and False means that the
sequence is too similar to a sequence in the test set and must be excluded.
## Method: `tabular()`

```python
tabular(self, columns: list[str]) -> DataFrame:
```

**Description:** Convert the sample to tabular data. Only these fields are supported:
- id: DBAASP ID
- sequence: noncanonical: O is Ornithine, B is DAB
- smiles: Note that only the first SMILES string is used
- nterminal: None or ACT
- cterminal: None or AMD
- targets <target_name>: Note that only the consensus value is used
- hc50: Note that only the consensus value is used

## Example:
```
df = dataset.tabular(["id", "sequence", "nterminal", "cterminal", "hc50", "Escherichia coli"])
print(df)
```

## Method: `with_bacterial_targets()`

```python
with_bacterial_targets(self, allowed: list[str]) -> DBAASPDataset:
```

**Description:** Keep only samples that have at least one bacterial target in the allowed list.

## Method: `with_bond()`

```python
with_bond(self, bond_type: list[typing.Literal[DSB, AMD, None]]) -> DBAASPDataset:
```

**Description:** Keep only samples that have bonds within the specified bond types. None means no bond is allowed.
For example, to keep only samples with disulfide bonds, use bond_type=['DSB']. To keep only samples with no
bonds, use bond_type=[None].

## Method: `with_canonical_only()`

```python
with_canonical_only(self) -> DBAASPDataset:
```

**Description:** Keep only samples that have only canonical amino acids in their sequence.
Sequences with non-canonical amino acids represented by the letters O (Ornithine) and B (DAB) or X are
filtered out.

## Method: `with_common_only()`

```python
with_common_only(self) -> DBAASPDataset:
```

**Description:** Keep only samples that have only common amino acids in their sequence: canonical amino acids and
Ornithin (O) and DAB (B).

## Method: `with_efficiency_below()`

```python
with_efficiency_below(self, threshold: float) -> DBAASPDataset:
```

**Description:** Keep only samples that have at least one bacterial target with efficiency below the given threshold (in µM).

## Method: `with_hc50()`

```python
with_hc50(self) -> DBAASPDataset:
```

**Description:** Keep only samples that have hemolytic activity (HC50) reported.

## Method: `with_l_aa_only()`

```python
with_l_aa_only(self) -> DBAASPDataset:
```

**Description:** Keep only samples that have only L-amino acids in their sequence.
D-amino acids are represented by lowercase letters.

## Method: `with_length_range()`

```python
with_length_range(self, min_length: typing.Union[int, NoneType], max_length: typing.Union[int, NoneType]) -> DBAASPDataset:
```

**Description:** Keep only samples that have a sequence length within the specified range.

**Parameters:**
- `min_length`: Minimum length of the sequence. If None, no minimum length is enforced.
- `max_length`: Maximum length of the sequence. If None, no maximum length is enforced.

## Method: `with_terminal_modification()`

```python
with_terminal_modification(self, nterminal: typing.Union[bool, NoneType], cterminal: typing.Union[bool, NoneType]) -> DBAASPDataset:
```

**Description:** Keep only samples that have the specified terminal modifications.

**Parameters:**
- `nterminal`: N-terminal modification to filter by. Use None for free N-terminus, or 'ACT' for acetylation. Use True to keep only sequence having 'ACT' modification, or False to keep only sequences with free N-terminus.
- `cterminal`: C-terminal modification to filter by. Use None for free C-terminus, or 'AMD' for amidation. Use True to keep only sequence having 'AMD' modification, or False to keep only sequences with free C-terminus.

