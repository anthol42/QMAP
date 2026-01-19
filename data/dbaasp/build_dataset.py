import json
from collections import defaultdict
import numpy as np
from dbaasp import Peptide, fetch_raw
from utils import min_or, max_or, get_iqr, classify_genus


not_mammals = [
    "Chicken",
    "Fish",
    "Tilapia",
    "Trout",
    "Salmon",
    "Cod",
    "Hybrid",
    "Croaker",
    "Hagfish",
    "Lizard",
    "Bird"
]

def build_dbaasp_dataset(out_path: str, cache_path: str = ".cache/DBAASP_raw.json"):
    raw_data = fetch_raw(cache_path, load_cache=True)

    dataset = []
    for sample in raw_data:
        peptide = Peptide(sample)

        # Step 1: Filtering
        if peptide.complexity != 'Monomer':
            continue

        if peptide.nTerminus is not None and peptide.nTerminus != 'ACT':
            continue

        if peptide.cTerminus is not None and peptide.cTerminus != 'AMD':
            continue

        if "X" in peptide.common_sequence.upper() and len(peptide.smiles) == 0:
            continue

        # Filter out invalid bond types
        if any(bond.type not in ["DSB", "AMD"] for bond in peptide.intrachainBonds):
            continue

        # Step 2: Extract bacteria labels
        targets = defaultdict(lambda: ([], []))
        if peptide.targets:
            for target in peptide.targets:
                # We keep only MIC measures
                if target.activityMeasureType != 'MIC':
                    continue
                # We keep only bacterial targets
                cls = classify_genus(target.specie.split(" ")[0])
                if cls != "Bacteria":
                    continue
                if target.unit is None:
                    continue

                # We keep track of all defined bounds (not <, >, or nan)
                minActs, maxActs = targets[target.specie]
                new_minAct, new_maxAct = target.minActivity, target.maxActivity
                if not np.isnan(new_minAct) and new_minAct != 0:
                    minActs.append(new_minAct)
                if not np.isnan(new_maxAct) and not np.isinf(new_maxAct):
                    maxActs.append(new_maxAct)
                targets[target.specie] = (minActs, maxActs)

        # We compute a consensus value by removing outliers using an IQR, and save the
        # full range (min, max, mean)
        consensus_targets = {}
        for target, (minActs, maxActs) in targets.items():
            min_val = min_or(minActs)
            max_val = max_or(maxActs)

            # If no minimum bound is defined, we set it to the maximum bound
            if min_val == 0:
                min_val = max_val

            # If no maximum bound is defined, we set it to the minimum bound
            if np.isinf(max_val):
                max_val = min_val

            # If no defined bounds were provided, we skip this target
            if len(minActs) + len(maxActs) == 0:
                continue

            all_val = minActs + maxActs
            q1, q3, iqr = get_iqr(all_val)
            valid = [x for x in all_val if q1 - 1.5 * iqr <= x <= q3 + 1.5 * iqr]
            mean = sum(valid) / len(valid)
            consensus_targets[target] = (min_val, max_val, mean)

        # Step 3: Extract hemolytic labels
        hemolytic_hc50 = None
        if peptide.toxicity and any("erythrocytes" in target.target for target in peptide.toxicity):
            hemolytic_targets = defaultdict(lambda: ([], []))
            for target in peptide.toxicity:
                # We keep only hemolytic targets (erythrocytes)
                if "erythrocytes" in target.target:
                    cls = target.target.split(" ")[0]
                    # We filter out all non-mammalian erythrocytes
                    if cls in not_mammals:
                        continue
                    # We keep HC50 measures only
                    if target.activityMeasureType != '50% Hemolysis':
                        continue

                    # Again, we keep track of all defined bounds (not <, >, or nan)
                    minActs, maxActs = hemolytic_targets[cls]
                    new_minAct, new_maxAct = target.minActivity, target.maxActivity
                    if not np.isnan(new_minAct) and new_minAct != 0:
                        minActs.append(new_minAct)
                    if not np.isnan(new_maxAct) and not np.isinf(new_maxAct):
                        maxActs.append(new_maxAct)
                    hemolytic_targets[cls] = (minActs, maxActs)

            if len(hemolytic_targets) == 0:
                hemolytic_hc50 = None
            else:
                # Then, prioritise human erythrocytes. If not present, take the consensus of all other
                # mammalian erythrocytes
                if "Human" in hemolytic_targets:
                    minActs, maxActs = hemolytic_targets["Human"]
                else:
                    minActs, maxActs = [], []
                    for cls, (cls_minAct, cls_maxAct) in hemolytic_targets.items():
                        minActs.extend(cls_minAct)
                        maxActs.extend(cls_maxAct)

                # If no defined bounds were provided, we skip this target
                if len(minActs) + len(maxActs) == 0:
                    hemolytic_hc50 = None
                else:
                    min_val = min_or(minActs)
                    max_val = max_or(maxActs)

                    # If no maximum bound is defined, we set it to the minimum bound
                    if np.isinf(max_val):
                        max_val = min_val

                    # If no minimum bound is defined, we set it to the maximum bound
                    if min_val == 0:
                        min_val = max_val

                    # We compute the consensus value by removing outliers using an IQR, and save the
                    # full range (min, max, mean)
                    all_val = minActs + maxActs
                    q1, q3, iqr = get_iqr(all_val)
                    valid = [x for x in all_val if q1 - 1.5 * iqr <= x <= q3 + 1.5 * iqr]
                    mean = sum(valid) / len(valid)
                    hemolytic_hc50 = (min_val, max_val, mean)

        # Step 4: Store peptide data
        dataset.append(dict(
            id=peptide.id,
            sequence=peptide.common_sequence,
            smiles=peptide.smiles,
            nterminal=peptide.nTerminus,
            cterminal=peptide.cTerminus,
            bonds=[(bond.start, bond.end, bond.type) for bond in peptide.intrachainBonds],
            targets=consensus_targets,
            hemolytic_hc50=hemolytic_hc50
        ))

    with open(out_path, "w") as f:
        json.dump(dataset, f)

if __name__ == '__main__':
    build_dbaasp_dataset("../build/dbaasp.json", cache_path="../.cache/DBAASP_raw.json")
