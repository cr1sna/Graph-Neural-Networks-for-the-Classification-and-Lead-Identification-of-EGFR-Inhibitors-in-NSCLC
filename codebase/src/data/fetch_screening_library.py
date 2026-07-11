"""
fetch_screening_library.py

Fetches a virtual screening library from ChEMBL by:
1. Pulling all small-molecule SMILES from ChEMBL that have:
   - Molecular Weight between 250-600 Da (drug-like)
   - LogP between -1 and 7
   - NOT previously tested on CHEMBL203 (EGFR) — i.e., truly unseen compounds
2. Deduplicates by canonical SMILES.
3. Saves to data/screening_library.csv

Strategy: Fetch compounds that have been tested on OTHER kinases (e.g., VEGFR2,
BRAF, CDK2) — these are drug-like kinase inhibitors that the EGFR model has
never seen, making the virtual screen realistic and scientifically meaningful.
"""

import os
import time
import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# ChEMBL target IDs for kinases other than EGFR (CHEMBL203)
# We'll pull candidates tested on these targets as our "unseen" library
KINASE_TARGETS = {
    "VEGFR2":  "CHEMBL279",
    "BRAF":    "CHEMBL5145",
    "CDK2":    "CHEMBL301",
    "ABL1":    "CHEMBL1862",
    "MET":     "CHEMBL2954",
}

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
SAVE_PATH   = "data/screening_library.csv"
MAX_PER_TARGET = 500    # compounds per kinase target
MW_MIN, MW_MAX = 250, 600
LOGP_MIN, LOGP_MAX = -1, 7


def fetch_active_smiles_for_target(target_id: str, max_records: int = 500) -> list[dict]:
    """Fetch active SMILES for a given ChEMBL target."""
    url = f"{CHEMBL_API}/activity.json"
    params = {
        "target_chembl_id":   target_id,
        "standard_type":      "IC50",
        "standard_relation":  "=",
        "pchembl_value__gte": 6.0,          # pIC50 >= 6 → active
        "molecule_chembl_id__isnull": False,
        "canonical_smiles__isnull": False,
        "limit": max_records,
        "offset": 0,
        "format": "json",
    }
    records = []
    while True:
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Warning: request failed for {target_id}: {e}")
            break

        activities = data.get("activities", [])
        for act in activities:
            smiles = act.get("canonical_smiles")
            cmpd_id = act.get("molecule_chembl_id")
            pchembl = act.get("pchembl_value")
            if smiles and cmpd_id:
                records.append({
                    "smiles":      smiles,
                    "chembl_id":   cmpd_id,
                    "source_target": target_id,
                    "pchembl_value": pchembl,
                })

        page_meta = data.get("page_meta", {})
        next_url  = page_meta.get("next")
        if not next_url or len(records) >= max_records:
            break
        params["offset"] += params["limit"]
        time.sleep(0.3)

    return records


def is_drug_like(smiles: str) -> tuple[bool, dict]:
    """Apply Lipinski Ro5 + TPSA filter, return (passes, props_dict)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, {}
    mw   = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot  = Descriptors.NumRotatableBonds(mol)
    props = {"MW": round(mw,2), "LogP": round(logp,2),
             "HBD": hbd, "HBA": hba, "TPSA": round(tpsa,2), "RotBonds": rot}
    passes = (MW_MIN <= mw <= MW_MAX and
              LOGP_MIN <= logp <= LOGP_MAX and
              hbd <= 5 and hba <= 10 and tpsa <= 140)
    return passes, props


def canonical(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def fetch_library(training_smiles_path: str = "data/processed/egfr_processed.csv") -> pd.DataFrame:
    """
    Main function: fetch, filter, deduplicate and return screening library.
    Already-seen SMILES (in the training set) are excluded.
    """
    # Load known training SMILES to exclude them
    seen_smiles = set()
    if os.path.exists(training_smiles_path):
        df_train = pd.read_csv(training_smiles_path)
        col = "canonical_smiles" if "canonical_smiles" in df_train.columns else "smiles"
        for s in df_train[col].dropna():
            c = canonical(s)
            if c:
                seen_smiles.add(c)
        print(f"Loaded {len(seen_smiles)} known training SMILES to exclude.")
    else:
        print("Warning: training SMILES file not found — no deduplication against training set.")

    all_records = []
    for name, tid in KINASE_TARGETS.items():
        print(f"  Fetching from {name} ({tid}) …")
        recs = fetch_active_smiles_for_target(tid, max_records=MAX_PER_TARGET)
        print(f"    → {len(recs)} raw records")
        all_records.extend(recs)
        time.sleep(0.5)

    # Canonicalise + filter drug-like + exclude training set
    filtered = []
    seen_canon = set(seen_smiles)
    for rec in all_records:
        raw = rec["smiles"]
        canon = canonical(raw)
        if canon is None or canon in seen_canon:
            continue
        passes, props = is_drug_like(canon)
        if not passes:
            continue
        seen_canon.add(canon)
        filtered.append({**rec, "canonical_smiles": canon, **props})

    df = pd.DataFrame(filtered)
    os.makedirs("data", exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)
    print(f"\nScreening library: {len(df)} unique drug-like compounds saved → {SAVE_PATH}")
    return df


if __name__ == "__main__":
    df = fetch_library()
    print(df[["chembl_id","canonical_smiles","MW","LogP","source_target"]].head(10))
