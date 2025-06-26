import re
from tqdm import tqdm
import requests
import os
from pathlib import PurePath
import pandas as pd
from bs4 import BeautifulSoup
from warnings import warn
from glob import glob

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

def fetch_list():
    """
    Return a dataframe containing all the datasets metadata, including the URLs to download their fasta
    """
    resp = requests.get('https://peptideatlas.org/builds/')
    if not resp.ok:
        raise Exception(f"Failed to fetch the dataset list. Status code: {resp.status_code}")
    html_content = resp.text
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find table by ID
    table = soup.find('table', {'id': "bdtable"})

    if table is None:
        raise ValueError(f"Table with id 'bdtable' not found")

    # Extract table data
    rows = []
    headers = []

    # Get headers (from thead or first tr)
    thead = table.find('thead')
    if thead:
        header_row = thead.find('tr')
    else:
        header_row = table.find('tr')

    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]

    # Get data rows (skip header row if it was the first tr)
    tbody = table.find('tbody')
    if tbody:
        data_rows = tbody.find_all('tr')
    else:
        all_rows = table.find_all('tr')
        # Skip first row if it contains headers
        data_rows = all_rows[1:] if not thead and headers else all_rows

    # Extract data from each row
    for row in data_rows:
        cells = row.find_all(['td', 'th'])
        row_data = []
        for cell in cells:
            # Check if cell contains a link ending with .fasta
            link = cell.find('a')
            if link and link.get('href'):
                text = link.get_text(strip=True)
                if text.endswith('.fasta'):
                    # Return markdown format link
                    href = link.get('href')
                    row_data.append(f"[{text}]({href})")
                else:
                    row_data.append(cell.get_text(strip=True))
            else:
                row_data.append(cell.get_text(strip=True))

        if row_data:  # Skip empty rows
            rows.append(row_data)

    # Create DataFrame
    if headers and rows:
        # Ensure all rows have same number of columns as headers
        max_cols = len(headers)
        rows = [row[:max_cols] + [''] * (max_cols - len(row)) for row in rows]
        df = pd.DataFrame(rows, columns=headers)
    elif rows:
        # No headers found, use default column names
        max_cols = max(len(row) for row in rows) if rows else 0
        headers = [f'Column_{i + 1}' for i in range(max_cols)]
        rows = [row[:max_cols] + [''] * (max_cols - len(row)) for row in rows]
        df = pd.DataFrame(rows, columns=headers)
    else:
        # Empty table
        df = pd.DataFrame()

    return df

def fetch_file(name, path):
    if not os.path.exists('.cache/files'):
        os.makedirs('.cache/files')
    if os.path.exists(f'.cache/files/{name}'):
        return f'.cache/files/{name}'
    else:
        url = f'https://peptideatlas.org/builds/{path}'
        resp = requests.get(url)
        if not resp.ok:
            warn(Warning(f"Failed to fetch the file {name}. Status code: {resp.status_code} at url {url}"))
        file_path = PurePath('.cache/files', name)
        with open(file_path, 'w') as f:
            f.write(resp.text)

def fetch_files(df: pd.DataFrame):
    """
    Download all files in the dataframe and return a new dataframe with the local paths
    """
    name_and_urls = df["Peptide Sequences"].tolist()

    parsed = [re.findall(r'(\[.*?])(\(.*?\))', s)[0] for s in name_and_urls if s]
    names = [e[0][1:-1] for e in parsed]
    urls = [e[1][1:-1] for e in parsed]
    for name, url in tqdm(zip(names, urls), total=len(names)):
        fetch_file(name, url)

def read_fasta(path):
    """
    Read a fasta file and return a dictionary with the sequence ID as key and the sequence as value.
    :param path: Path to the fasta file
    :return: Dictionary with sequence ID as key and sequence as value
    """
    sequences = {}
    with open(path, 'r') as f:
        current_id = None
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                current_id = line[1:]  # Remove '>'
            else:
                if current_id is not None:
                    sequences[current_id] = line
    return sequences

def make_dataset():
    """
    Create a dataset by mergin all fasta files into a single file and removes duplicates.
    :return: None
    """
    all_sequences = set()
    for file in glob('.cache/files/*.fasta'):
        sequences = read_fasta(file)
        all_sequences.update(sequences.values())

    # Filter sequences such that they are smaller than 50 amino acids
    all_sequences = {seq for seq in all_sequences if len(seq) <= 50}

    # Write to a new fasta file
    with open('.cache/peptide_atlas.fasta', 'w') as f:
        for i, sequence in enumerate(all_sequences):
            f.write(f'>seq_{i}\n{sequence}\n')

    print("Total dataset size:", len(all_sequences))
if __name__ == '__main__':
    builds = fetch_list()[["Build Name", "Peptide Sequences"]]
    # Keep only most recents
    builds = builds.loc[builds['Build Name'] != ""]
    fetch_files(builds)
    make_dataset()

