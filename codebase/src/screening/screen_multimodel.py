"""screen_multimodel.py — Multi-model virtual screening of Nepali plant compounds.

Rather than relying on a single (and, as the benchmark showed, the weakest)
classifier, this module deploys the study's two *best-performing* models for the
prospective Nepali medicinal plant screen:

* the best graph neural network — **GCN** — operating on the 29-dim node /
  12-dim edge molecular graphs, and
* the best overall classifier — **Random Forest** — operating on 2048-bit
  ECFP4 fingerprints.

Both models are (re)trained on the identical Bemis-Murcko scaffold-split training
partition used for benchmarking, then applied to the 535-compound COCONUT 2.0
library. Each compound receives a predicted activity probability from both models,
enabling consensus (agreement) analysis. Two ranked lead CSVs are written:
``results/nepali_leads_gcn.csv`` and ``results/nepali_leads_rf.csv``.

Run with::

    python -m src.screening.screen_multimodel
"""
from __future__ import annotations
import logging, pathlib, random
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.data.graph_dataset import MoleculeDataset, smiles_to_data
from src.data.scaffold_split import scaffold_split
from src.models.gnn_models import GCNModel
from src.models.baseline_models import RandomForestBaseline
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

PLANT_CSV   = pathlib.Path("data/nepali_plants/nepali_compounds.csv")
CLEANED_CSV = pathlib.Path("data/processed/egfr_cleaned.csv")
PYG_CACHE   = pathlib.Path("data/pyg_cache")
GCN_CKPT    = pathlib.Path("results/best_gcn.pt")
OUT_GCN     = pathlib.Path("results/nepali_leads_gcn.csv")
OUT_RF      = pathlib.Path("results/nepali_leads_rf.csv")
SEED        = 42


def _lipinski(row: pd.Series) -> bool:
    return (row.mw <= 500 and row.logp <= 5 and row.hbd <= 5 and row.hba <= 10)


def _finalise(lib: pd.DataFrame, probs: np.ndarray, out: pathlib.Path) -> pd.DataFrame:
    """Attach probabilities, Ro5 flag, rank, and write the ranked CSV."""
    df = lib.copy()
    df["P_active"] = np.round(probs, 4)
    df["ro5_compliant"] = df.apply(_lipinski, axis=1)
    df = df.sort_values("P_active", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Wrote %d ranked compounds to %s", len(df), out)
    return df


def screen_multimodel(
    plant_csv: pathlib.Path = PLANT_CSV,
    cleaned_csv: pathlib.Path = CLEANED_CSV,
    epochs: int = 80,
) -> dict:
    """Train GCN + Random Forest on the scaffold-split train set and screen the
    Nepali plant library with both. Returns a dict of the two ranked DataFrames."""
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    # --- data + identical scaffold split as the benchmark ---
    ds = MoleculeDataset(root=str(PYG_CACHE), csv_path=str(cleaned_csv))
    data_list = [ds[i] for i in range(len(ds))]
    train, val, _test = scaffold_split(data_list, 0.8, 0.1, 0.1, seed=SEED)

    # --- best GNN: GCN ---
    gcn = GCNModel()
    trainer = Trainer(batch_size=64, patience=15)
    trainer.train(gcn, train, val, epochs=epochs)
    torch.save(gcn.state_dict(), GCN_CKPT)

    # --- best classifier: Random Forest on ECFP4 ---
    rf = RandomForestBaseline()
    rf.fit([d.smiles for d in train], [int(d.y.item()) for d in train])

    # --- library ---
    lib = pd.read_csv(plant_csv)
    smiles = lib["smiles"].astype(str).tolist()

    # GCN inference on molecular graphs (dropping unparseable SMILES)
    graphs, keep = [], []
    for i, smi in enumerate(smiles):
        g = smiles_to_data(smi, 0, 0.0)
        if g is not None:
            graphs.append(g); keep.append(i)
    gcn.eval()
    gcn_probs = []
    for batch in DataLoader(graphs, batch_size=64, shuffle=False):
        with torch.no_grad():
            logits = gcn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            gcn_probs.extend(torch.sigmoid(logits.squeeze(1)).tolist())
    df_gcn = _finalise(lib.iloc[keep].reset_index(drop=True),
                       np.asarray(gcn_probs), OUT_GCN)

    # Random Forest inference on fingerprints (all compounds)
    rf_probs = rf.predict_proba(smiles)
    df_rf = _finalise(lib, np.asarray(rf_probs), OUT_RF)

    return {"gcn": df_gcn, "rf": df_rf}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    res = screen_multimodel()
    for name, df in res.items():
        n5 = int((df.P_active >= 0.5).sum())
        n9 = int((df.P_active >= 0.9).sum())
        logger.info("%s: %d actives (p>=0.5), %d high-confidence (p>=0.9); top = %s",
                    name.upper(), n5, n9, df.iloc[0].compound_name)


if __name__ == "__main__":
    main()
