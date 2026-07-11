"""eda.py — Exploratory data analysis of the cleaned EGFR dataset.

Runs descriptive analysis on the cleaned ChEMBL EGFR dataset *before* molecular
graphs are constructed, characterising class balance, the pIC50 distribution,
physicochemical property ranges, and Bemis-Murcko scaffold diversity. A summary
is logged to the console and written to ``results/eda_summary.json``; the
accompanying EDA figures are produced by ``src.utils.make_thesis_figures``.

Usage::

    python -m src.data.eda
"""
from __future__ import annotations

import json
import logging
import pathlib

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CSV: pathlib.Path = pathlib.Path("data/processed/egfr_cleaned.csv")
DEFAULT_OUT: pathlib.Path = pathlib.Path("results/eda_summary.json")


def run_eda(
    csv_path: pathlib.Path | str = DEFAULT_CSV,
    output_file: pathlib.Path | str = DEFAULT_OUT,
    property_sample: int = 4000,
) -> dict:
    """Compute and report exploratory statistics of the cleaned dataset.

    Args:
        csv_path: Path to the cleaned CSV (columns: chembl_id, smiles, pIC50, label).
        output_file: Destination JSON file for the summary statistics.
        property_sample: Number of molecules to sample for RDKit property stats
            (the full set is used for class balance, pIC50, and scaffolds).

    Returns:
        Dictionary of summary statistics.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = pathlib.Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {csv_path}. Run preprocess.py first."
        )

    df = pd.read_csv(csv_path)
    required = {"smiles", "pIC50", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    n = len(df)
    n_active = int((df["label"] == 1).sum())
    n_inactive = n - n_active

    # pIC50 statistics
    pic50 = df["pIC50"].astype(float)

    # Physicochemical properties on a random sample (for speed)
    sample = df.sample(min(property_sample, n), random_state=0)
    mw, logp = [], []
    for smi in sample["smiles"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is not None:
            mw.append(Descriptors.MolWt(mol))
            logp.append(Descriptors.MolLogP(mol))
    mw, logp = np.array(mw), np.array(logp)

    # Bemis-Murcko scaffold diversity over the full set
    scaffolds: dict[str, int] = {}
    for smi in df["smiles"]:
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            key = Chem.MolToSmiles(
                MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=False
            )
            scaffolds[key] = scaffolds.get(key, 0) + 1
        except Exception:  # noqa: BLE001
            continue
    sizes = sorted(scaffolds.values(), reverse=True)
    n_scaffolds = len(sizes)
    n_singletons = sum(1 for s in sizes if s == 1)

    summary = {
        "n_compounds": n,
        "n_active": n_active,
        "n_inactive": n_inactive,
        "active_fraction": round(n_active / n, 4),
        "pIC50_min": round(float(pic50.min()), 2),
        "pIC50_median": round(float(pic50.median()), 2),
        "pIC50_mean": round(float(pic50.mean()), 2),
        "pIC50_max": round(float(pic50.max()), 2),
        "mw_mean": round(float(mw.mean()), 1),
        "mw_pct_le_500": round(100 * float((mw <= 500).mean()), 1),
        "logp_mean": round(float(logp.mean()), 2),
        "logp_pct_le_5": round(100 * float((logp <= 5).mean()), 1),
        "n_unique_scaffolds": n_scaffolds,
        "n_singleton_scaffolds": n_singletons,
        "singleton_fraction": round(n_singletons / max(n_scaffolds, 1), 4),
        "largest_scaffold_size": sizes[0] if sizes else 0,
    }

    # ------------------------------------------------------------------ report
    logger.info("─── EGFR Dataset Exploratory Analysis ────────────────────")
    logger.info("Compounds          : %d", summary["n_compounds"])
    logger.info("Active / Inactive  : %d / %d (%.1f%% active)",
                n_active, n_inactive, 100 * summary["active_fraction"])
    logger.info("pIC50 (min/med/max): %.2f / %.2f / %.2f",
                summary["pIC50_min"], summary["pIC50_median"], summary["pIC50_max"])
    logger.info("MW mean            : %.1f Da (%.1f%% <= 500)",
                summary["mw_mean"], summary["mw_pct_le_500"])
    logger.info("LogP mean          : %.2f (%.1f%% <= 5)",
                summary["logp_mean"], summary["logp_pct_le_5"])
    logger.info("Unique scaffolds   : %d (%.1f%% singletons)",
                n_scaffolds, 100 * summary["singleton_fraction"])
    logger.info("Largest scaffold   : %d compounds", summary["largest_scaffold_size"])
    logger.info("──────────────────────────────────────────────────────────")

    output_file = pathlib.Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("EDA summary saved to %s", output_file)

    return summary


if __name__ == "__main__":
    run_eda()
