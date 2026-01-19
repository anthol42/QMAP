# Prepare the data
1. Sync the repository using uv:
```bash
uv sync
```
## DBAASP
1. Run the file (Download, extract and format the DBAASP database into a ML dataset): `build_dataset.py`
```shell
uv run dbaasp/build_dataset.py
```
2. Run the file (Split the dataset into five splits for the benchmark): `make_benchmark.py`
```shell
uv run dbaasp/make_benchmark.py
```

## Peptide Atlas
Download the peptide atlas database by running the `pepAtlas/download_pep_atlas.py` script:
```bash
uv run download_pep_atlas.py
```

## Maxmimum identity experiment
To run the maximum identity experiment, execute the following command:
```bash
uv run pep_atlas2dbaasp.py
```

> **Note:**<br>
> It takes around 8h on a MacBook pro M4 to run. Once it has run, the results will be cached, so subsequent runs will 
> be much faster.