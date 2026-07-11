"""screen_nepali.py — Virtual screening of Nepali medicinal plant compounds.

Loads the trained AttentiveFP model, converts plant compound SMILES to
molecular graphs using the identical 29-dim node + 12-dim edge featurisation
as the training data, runs inference, applies Lipinski Ro5 filtering, and
outputs a ranked lead candidate CSV.
"""
from __future__ import annotations
import logging, pathlib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from torch_geometric.loader import DataLoader

from src.data.graph_dataset import smiles_to_data
from src.models.gnn_models import AttentiveFPModel

logger = logging.getLogger(__name__)

PLANT_CSV    = pathlib.Path("data/nepali_plants/nepali_compounds.csv")
CHECKPOINT   = pathlib.Path("results/best_attentivefp.pt")
OUTPUT_FILE  = pathlib.Path("results/nepali_leads.csv")


def _lipinski_compliant(row: pd.Series) -> bool:
    """Return True if compound satisfies all Lipinski Ro5 criteria."""
    return (
        row["mw"]   <= 500 and
        row["logp"] <= 5   and
        row["hbd"]  <= 5   and
        row["hba"]  <= 10
    )


def screen_nepali_compounds(
    plant_csv: pathlib.Path = PLANT_CSV,
    checkpoint: pathlib.Path = CHECKPOINT,
    output_file: pathlib.Path = OUTPUT_FILE,
    model_kwargs: dict | None = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Score Nepali medicinal plant compounds with the trained AttentiveFP.

    Args:
        plant_csv: Path to nepali_compounds.csv from fetch_nepali_plants.py.
        checkpoint: Path to best_attentivefp.pt checkpoint.
        output_file: Destination ranked leads CSV.
        model_kwargs: AttentiveFP constructor kwargs (must match training).
        batch_size: Inference batch size.

    Returns:
        DataFrame ranked by P(active) with Ro5 and ADMET-proxy columns.

    Raises:
        FileNotFoundError: If plant_csv or checkpoint are missing.
        ValueError: If plant_csv contains no valid molecules.
    """
    for p in (plant_csv, checkpoint):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    logger.info("Loading Nepali plant compounds from %s …", plant_csv)
    df = pd.read_csv(plant_csv)

    # Convert SMILES to PyG Data objects
    data_list = []
    valid_rows = []
    for _, row in df.iterrows():
        smi = str(row["smiles"])
        data = smiles_to_data(smi, label=0, pic50=0.0)   # label=0 placeholder
        if data is not None:
            data_list.append(data)
            valid_rows.append(row)

    if not data_list:
        raise ValueError("No valid molecules found in plant compound CSV.")

    logger.info("Converted %d / %d molecules to graphs.", len(data_list), len(df))

    # Load model
    kwargs = model_kwargs or {}
    model = AttentiveFPModel(**kwargs)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()
    logger.info("Model loaded from %s (%d params).", checkpoint, model.get_num_params())

    # Inference
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    all_probs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            all_probs.extend(probs)

    # Assemble results
    result_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    result_df["P_active"] = np.round(all_probs, 6)
    result_df["ro5_compliant"] = result_df.apply(_lipinski_compliant, axis=1)

    # Rank by P(active) descending
    result_df = result_df.sort_values("P_active", ascending=False).reset_index(drop=True)
    result_df.insert(0, "rank", result_df.index + 1)

    # Select output columns
    out_cols = ["rank", "compound_name", "plant_species", "smiles",
                "P_active", "mw", "logp", "qed", "tpsa", "ro5_compliant"]
    result_df = result_df[[c for c in out_cols if c in result_df.columns]]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_file, index=False)
    logger.info("Ranked leads saved to %s", output_file)

    # Console table — top 10
    top10 = result_df.head(10)
    print("\n─── Top 10 Nepali Plant Lead Candidates ────────────────────────────")
    print(f"{'Rank':>4}  {'Compound':<28}  {'Plant Species':<25}  "
          f"{'P(active)':>9}  {'MW':>7}  {'LogP':>6}  {'QED':>6}  {'Ro5':>5}")
    print("─" * 100)
    for _, r in top10.iterrows():
        print(f"{int(r['rank']):>4}  {str(r.get('compound_name','?'))[:28]:<28}  "
              f"{str(r.get('plant_species','?'))[:25]:<25}  "
              f"{r['P_active']:>9.4f}  {r.get('mw',0):>7.1f}  "
              f"{r.get('logp',0):>6.2f}  {r.get('qed',0):>6.3f}  "
              f"{'Yes' if r.get('ro5_compliant') else 'No':>5}")
    print("─" * 100)
    return result_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    screen_nepali_compounds()
