"""fetch_nepali_plants.py — Build Nepali medicinal plant compound library.

Loads the COCONUT 2.0 local CSV, filters by target Nepali/Himalayan plant
species, applies identical SMILES standardisation as preprocess.py, computes
ADMET-proxy physicochemical properties via RDKit, and saves a compound CSV.
"""
from __future__ import annotations
import logging, pathlib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Chem.SaltRemover import SaltRemover

logger = logging.getLogger(__name__)

COCONUT_CSV   = pathlib.Path("data/coconut/coconut_complete.csv")
OUTPUT_DIR    = pathlib.Path("data/nepali_plants")
OUTPUT_FILE   = OUTPUT_DIR / "nepali_compounds.csv"
# Curated fallback library of documented phytochemicals from the seven target
# species, used when the full COCONUT 2.0 download is unavailable. Columns:
# compound_name, plant_species, smiles.
SEED_CSV      = OUTPUT_DIR / "nepali_seed_compounds.csv"
MW_UPPER      = 1000.0

TARGET_SPECIES: list[str] = [
    "Swertia chirayita",
    "Rhododendron arboreum",
    "Nardostachys jatamansi",
    "Terminalia chebula",
    "Berberis aristata",
    "Ocimum sanctum",
    "Tinospora cordifolia",
]


def _clean_and_compute(smiles: str) -> dict | None:
    """Standardise SMILES and compute physicochemical properties.

    Args:
        smiles: Raw SMILES from COCONUT.

    Returns:
        Dict with keys (smiles, mw, logp, hbd, hba, tpsa, qed), or None
        if the molecule is invalid or exceeds MW limit.
    """
    if not smiles or pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        remover = SaltRemover()
        mol = remover.StripMol(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        mw = Descriptors.MolWt(mol)
        if mw > MW_UPPER:
            return None
        canon = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        return {
            "smiles": canon,
            "mw":     round(mw, 3),
            "logp":   round(Descriptors.MolLogP(mol), 3),
            "hbd":    rdMolDescriptors.CalcNumHBD(mol),
            "hba":    rdMolDescriptors.CalcNumHBA(mol),
            "tpsa":   round(Descriptors.TPSA(mol), 3),
            "qed":    round(QED.qed(mol), 4),
        }
    except Exception as exc:
        logger.debug("Skipping molecule '%s': %s", smiles, exc)
        return None


def _rows_from_coconut(coconut_csv: pathlib.Path) -> list[dict]:
    """Extract standardised compound rows from a COCONUT 2.0 CSV download.

    Args:
        coconut_csv: Path to the local COCONUT 2.0 CSV download.

    Returns:
        List of row dicts (compound_name, plant_species, + computed properties)
        for compounds matching any target species.

    Raises:
        ValueError: If the organism or SMILES column cannot be located.
    """
    logger.info("Loading COCONUT 2.0 from %s …", coconut_csv)
    df_raw = pd.read_csv(coconut_csv, low_memory=False)

    # Detect organism column
    org_col = None
    for candidate in ("organism_name", "source_organism", "organisms", "species"):
        if candidate in df_raw.columns:
            org_col = candidate
            break
    if org_col is None:
        raise ValueError(
            f"Cannot find organism column in COCONUT CSV. "
            f"Columns present: {list(df_raw.columns)}"
        )

    # Detect name and SMILES columns
    name_col = next((c for c in ("name", "compound_name", "identifier") if c in df_raw.columns), None)
    smi_col  = next((c for c in ("smiles", "canonical_smiles", "SMILES") if c in df_raw.columns), None)
    if smi_col is None:
        raise ValueError("Cannot find SMILES column in COCONUT CSV.")

    rows: list[dict] = []
    for species in TARGET_SPECIES:
        mask = df_raw[org_col].str.contains(species, case=False, na=False)
        subset = df_raw[mask].copy()
        logger.info("  %s: %d raw entries found.", species, len(subset))

        for _, row in subset.iterrows():
            props = _clean_and_compute(str(row[smi_col]))
            if props is None:
                continue
            cname = str(row[name_col]) if name_col and name_col in row else "Unknown"
            rows.append({"compound_name": cname, "plant_species": species, **props})
    return rows


def _rows_from_seed(seed_csv: pathlib.Path) -> list[dict]:
    """Build standardised compound rows from the curated seed library.

    The seed library is a small, hand-verified set of documented phytochemicals
    from the target species, used when the full COCONUT 2.0 download is absent.

    Args:
        seed_csv: Path to the curated seed CSV (compound_name, plant_species,
            smiles columns).

    Returns:
        List of row dicts identical in schema to :func:`_rows_from_coconut`.

    Raises:
        FileNotFoundError: If the seed CSV is missing.
    """
    if not seed_csv.exists():
        raise FileNotFoundError(f"Seed compound library not found at {seed_csv}.")

    logger.info("Building library from curated seed file %s …", seed_csv)
    df_seed = pd.read_csv(seed_csv)
    rows: list[dict] = []
    for _, row in df_seed.iterrows():
        props = _clean_and_compute(str(row["smiles"]))
        if props is None:
            logger.warning("Seed compound failed standardisation: %s", row.get("compound_name"))
            continue
        rows.append({
            "compound_name": str(row["compound_name"]),
            "plant_species": str(row["plant_species"]),
            **props,
        })
    return rows


def fetch_nepali_plant_compounds(
    coconut_csv: pathlib.Path = COCONUT_CSV,
    output_file: pathlib.Path = OUTPUT_FILE,
    seed_csv: pathlib.Path = SEED_CSV,
) -> pd.DataFrame:
    """Build the Nepali medicinal plant compound library and compute properties.

    Primary source is a local COCONUT 2.0 download filtered to the seven target
    species. If that download is unavailable, the function falls back to the
    curated seed library so the downstream screening pipeline remains runnable.

    Args:
        coconut_csv: Path to local COCONUT 2.0 CSV download (preferred source).
        output_file: Destination CSV path.
        seed_csv: Curated fallback library used when ``coconut_csv`` is absent.

    Returns:
        DataFrame with columns [compound_name, plant_species, smiles,
        mw, logp, hbd, hba, tpsa, qed].

    Raises:
        FileNotFoundError: If neither the COCONUT download nor the seed library
            is available.
        ValueError: If no valid compounds are produced from the chosen source.
    """
    if coconut_csv.exists():
        rows = _rows_from_coconut(coconut_csv)
    else:
        logger.warning(
            "COCONUT 2.0 download not found at %s — falling back to the curated "
            "seed library. For the full library, download from "
            "https://zenodo.org/records/13382751 and place at %s.",
            coconut_csv, coconut_csv,
        )
        rows = _rows_from_seed(seed_csv)

    if not rows:
        raise ValueError(
            "No valid compounds found for any target Nepali plant species. "
            "Check the COCONUT CSV / seed library and organism column format."
        )

    df = pd.DataFrame(rows)
    # Deduplicate by canonical SMILES (keep first occurrence per species)
    df = df.drop_duplicates(subset=["smiles"]).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    logger.info("─── Nepali Plant Library Summary ────────────────────────")
    for sp, grp in df.groupby("plant_species"):
        logger.info("  %-35s : %d compounds", sp, len(grp))
    logger.info("  %-35s : %d compounds", "TOTAL", len(df))
    logger.info("Saved to %s", output_file)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetch_nepali_plant_compounds()
