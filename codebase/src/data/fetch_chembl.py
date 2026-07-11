"""fetch_chembl.py — Retrieve EGFR IC50 bioactivity data from ChEMBL.

This module queries the ChEMBL REST API for IC50 bioactivity measurements
against the EGFR target (CHEMBL203) and saves the raw results to disk.
IC50 values reported in ChEMBL for this target are in nanomolar (nM) units,
as assumed throughout the downstream pIC50 conversion.
"""

from __future__ import annotations

import logging
import pathlib

import pandas as pd
from chembl_webresource_client.new_client import new_client

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
TARGET_CHEMBL_ID: str = "CHEMBL203"
DEFAULT_OUTPUT_DIR: pathlib.Path = pathlib.Path("data/raw")
DEFAULT_OUTPUT_FILE: str = "egfr_chembl_raw.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_egfr_data(
    output_dir: pathlib.Path | str = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """Fetch EGFR IC50 bioactivity records from ChEMBL and save to CSV.

    Queries ChEMBL (target CHEMBL203) for all IC50 measurements with an
    exact standard relation (``=``). IC50 values are assumed to be in
    nanomolar (nM) units as provided by ChEMBL for this target.

    Args:
        output_dir: Directory in which to save the raw CSV file.
            Created automatically if it does not exist.

    Returns:
        A :class:`pandas.DataFrame` containing all retrieved bioactivity
        records.

    Raises:
        RuntimeError: If the ChEMBL API returns zero records.
        OSError: If the output directory cannot be created or the file
            cannot be written.
    """
    output_dir = pathlib.Path(output_dir)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.error("Cannot create output directory %s: %s", output_dir, exc)
        raise

    output_file: pathlib.Path = output_dir / DEFAULT_OUTPUT_FILE
    logger.info("Fetching IC50 data for target %s from ChEMBL …", TARGET_CHEMBL_ID)

    try:
        activity = new_client.activity
        res = activity.filter(
            target_chembl_id=TARGET_CHEMBL_ID,
            standard_type="IC50",
            standard_relation="=",
        )
        df = pd.DataFrame.from_dict(res)
    except Exception as exc:
        logger.error("ChEMBL API request failed: %s", exc)
        raise RuntimeError(f"ChEMBL API request failed: {exc}") from exc

    if df.empty:
        raise RuntimeError(
            f"ChEMBL returned zero records for target {TARGET_CHEMBL_ID}. "
            "Check your network connection or the target ID."
        )

    logger.info("Retrieved %d records. Saving to %s …", len(df), output_file)

    try:
        df.to_csv(output_file, index=False)
    except OSError as exc:
        logger.error("Failed to write CSV to %s: %s", output_file, exc)
        raise

    logger.info("Raw data saved to %s", output_file)
    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fetch_egfr_data()
