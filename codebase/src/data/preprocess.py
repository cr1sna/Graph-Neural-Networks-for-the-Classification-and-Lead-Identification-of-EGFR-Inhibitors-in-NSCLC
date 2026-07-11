"""preprocess.py — Clean and standardise raw ChEMBL EGFR bioactivity data.

Converts raw IC50 values (nM) to pIC50, applies binary activity labels,
removes salts, canonicalises SMILES, filters by molecular weight, and
deduplicates by median pIC50 per unique canonical SMILES.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.SaltRemover import SaltRemover

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INPUT_FILE: pathlib.Path = pathlib.Path("data/raw/egfr_chembl_raw.csv")
DEFAULT_OUTPUT_FILE: pathlib.Path = pathlib.Path("data/processed/egfr_cleaned.csv")
PIC50_THRESHOLD: float = 6.0          # pIC50 >= 6.0  →  active (IC50 <= 1 µM)
MW_UPPER_LIMIT: float = 1000.0        # Da


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def clean_smiles(smiles: str) -> Optional[Tuple[str, float]]:
    """Standardise a SMILES string using RDKit.

    Performs salt removal (keeps the largest fragment), canonicalises the
    SMILES representation, and returns the molecular weight.

    Args:
        smiles: Input SMILES string (may be messy / contain salts).

    Returns:
        A ``(canonical_smiles, molecular_weight)`` tuple, or ``None`` if
        the SMILES is invalid, empty after salt removal, or causes an
        RDKit exception.
    """
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.debug("RDKit could not parse SMILES: %s", smiles)
            return None

        remover = SaltRemover()
        mol = remover.StripMol(mol)

        if mol is None or mol.GetNumAtoms() == 0:
            logger.debug("Molecule empty after salt removal: %s", smiles)
            return None

        mw: float = Descriptors.MolWt(mol)
        canonical: str = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return canonical, mw

    except Exception as exc:  # noqa: BLE001
        logger.warning("Unexpected error cleaning SMILES '%s': %s", smiles, exc)
        return None


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def preprocess_data(
    input_file: pathlib.Path | str = DEFAULT_INPUT_FILE,
    output_file: pathlib.Path | str = DEFAULT_OUTPUT_FILE,
) -> pd.DataFrame:
    """Preprocess raw ChEMBL EGFR IC50 data into a clean ML-ready dataset.

    Pipeline steps:
        1. Load raw CSV produced by ``fetch_chembl.py``.
        2. Retain only exact IC50 measurements (standard_relation == ``=``).
        3. Drop rows with missing ``standard_value`` or ``canonical_smiles``.
        4. Remove zero / negative IC50 values.
        5. Convert IC50 (nM) → pIC50 using ``pIC50 = -log10(IC50 * 1e-9)``.
        6. Assign binary label: 1 (active) if pIC50 >= 6.0, else 0 (inactive).
        7. Clean SMILES: salt removal, canonicalisation.
        8. Filter: molecular weight <= 1000 Da.
        9. Deduplicate by canonical SMILES using **median** pIC50.
        10. Re-compute label from deduplicated pIC50.
        11. Save to CSV.

    Args:
        input_file: Path to the raw CSV file from ``fetch_chembl.py``.
        output_file: Path for the cleaned output CSV.

    Returns:
        A :class:`pandas.DataFrame` with columns
        ``['chembl_id', 'smiles', 'pIC50', 'label']``.

    Raises:
        FileNotFoundError: If ``input_file`` does not exist.
        ValueError: If required columns are missing from the input file,
            or if the cleaned dataset is empty.
        OSError: If the output file cannot be written.
    """
    input_file = pathlib.Path(input_file)
    output_file = pathlib.Path(output_file)

    # ------------------------------------------------------------------ load
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}. "
            "Run fetch_chembl.py first."
        )

    logger.info("Loading raw data from %s …", input_file)
    df = pd.read_csv(input_file)

    required_columns = {"standard_type", "standard_relation", "standard_value",
                        "canonical_smiles", "molecule_chembl_id"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing required columns: {missing}. "
            f"Columns present: {list(df.columns)}"
        )

    initial_len: int = len(df)
    logger.info("Loaded %d raw records.", initial_len)

    # -------------------------------------------------------- filter IC50 exact
    df = df[
        (df["standard_type"] == "IC50") & (df["standard_relation"] == "=")
    ].copy()
    logger.info("%d records after IC50 exact-value filter.", len(df))

    # ------------------------------------------------------- numeric & non-null
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value", "canonical_smiles"])
    df = df[df["standard_value"] > 0]
    logger.info("%d records after dropping missing/zero values.", len(df))

    # ---------------------------------------------------------- pIC50 & label
    df["pIC50"] = -np.log10(df["standard_value"] * 1e-9)
    df["label"] = (df["pIC50"] >= PIC50_THRESHOLD).astype(int)

    # --------------------------------------------------------- SMILES cleaning
    logger.info("Cleaning SMILES strings …")
    cleaned = df["canonical_smiles"].apply(clean_smiles)
    df["clean_smiles"] = [r[0] if r else None for r in cleaned]
    df["mw"] = [r[1] if r else None for r in cleaned]

    df = df.dropna(subset=["clean_smiles", "mw"])
    df = df[df["mw"] <= MW_UPPER_LIMIT]
    logger.info("%d records after SMILES cleaning and MW filter.", len(df))

    # --------------------------------------------------------- deduplication
    # Group duplicate canonical SMILES; use MEDIAN pIC50 across replicates.
    df = (
        df.groupby("clean_smiles")
        .agg(
            molecule_chembl_id=("molecule_chembl_id", "first"),
            pIC50=("pIC50", "median"),
        )
        .reset_index()
    )
    # Re-compute binary label from the deduplicated median pIC50
    df["label"] = (df["pIC50"] >= PIC50_THRESHOLD).astype(int)

    df = df.rename(columns={"clean_smiles": "smiles",
                             "molecule_chembl_id": "chembl_id"})
    df = df[["chembl_id", "smiles", "pIC50", "label"]]

    if df.empty:
        raise ValueError(
            "Cleaned dataset is empty — no compounds survived filtering. "
            "Check the raw data or filter thresholds."
        )

    # ------------------------------------------------------------------- save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(output_file, index=False)
    except OSError as exc:
        logger.error("Failed to write output CSV to %s: %s", output_file, exc)
        raise

    n_active = int(df["label"].sum())
    n_inactive = len(df) - n_active
    logger.info("Initial records:                    %d", initial_len)
    logger.info("Final records after cleaning:       %d", len(df))
    logger.info("Active compounds  (label=1): %d (%.1f%%)",
                n_active, 100 * n_active / len(df))
    logger.info("Inactive compounds (label=0): %d (%.1f%%)",
                n_inactive, 100 * n_inactive / len(df))
    logger.info("Processed data saved to %s", output_file)

    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    preprocess_data()
