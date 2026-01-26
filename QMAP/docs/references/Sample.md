*src.qmap.benchmark*
# Class: `Sample`

**Description:** No documentation available

## Method: `tabular()`

```python
tabular(self, columns: list[str]) -> list[str]:
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
sample = ...
columns = ["id", "sequence", "nterminal", "cterminal", "hc50", "Escherichia coli"]
print(sample.tabular(columns))
```

