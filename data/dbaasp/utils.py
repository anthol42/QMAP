import numpy as np
from ete4 import NCBITaxa

ncbi = NCBITaxa()

def min_or(values, default=0.):
    if len(values) == 0:
        return default
    return min(values)

def max_or(values, default=float('inf')):
    if len(values) == 0:
        return default
    return max(values)

def get_iqr(values):
    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    return q1, q3, iqr

def classify_genus(genus):
    try:
        taxid = ncbi.get_name_translator([genus])[genus][0]
        lineage = ncbi.get_lineage(taxid)
        names = ncbi.get_taxid_translator(lineage).values()

        if "Bacteria" in names:
            return "Bacteria"
        if "Archaea" in names:
            return "Archaea"
        if "Viruses" in names:
            return "Virus"
        if "Fungi" in names:
            return "Fungus"
        if "Metazoa" in names:
            return "Animal"
        if "Viridiplantae" in names:
            return "Plant"
        return "Other"
    except KeyError:
        return "Unknown"