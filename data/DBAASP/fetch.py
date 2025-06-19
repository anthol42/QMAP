import requests
from tqdm import tqdm
import json
import os

def fetch_peptide(identifier: int):
    url = f'https://dbaasp.org/peptides/{identifier}'
    res = requests.get(url, headers={
        'accept': 'application/json'
    })
    if not res.ok:
        print(f"An error happeneded while trying to fetch the data for peptide {identifier}.  \n"
                           f"The url was: {url} -status: {res.status_code}")
        return None

    return res.json()

def fetch_raw(out_path: str = '.cache/DBAASP_raw.json', load_cache: bool = True):
    if load_cache and os.path.exists(out_path):
        with open(out_path, 'r') as f:
            return json.load(f)

    MAX_ID = 23951 # As of June 2025

    raw_data = []
    for ID in tqdm(range(1, MAX_ID + 1)):
        pep = fetch_peptide(ID)
        if pep is not None:
            raw_data.append(pep)

    if not os.path.exists(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(raw_data, f)

    return raw_data