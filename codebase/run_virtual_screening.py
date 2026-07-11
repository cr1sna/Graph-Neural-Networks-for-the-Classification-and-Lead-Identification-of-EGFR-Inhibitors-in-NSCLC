"""
run_virtual_screening.py

End-to-end virtual screening pipeline using the tuned AttentiveFP model:
  1. Fetch a drug-like screening library from ChEMBL (unseen kinase inhibitors)
  2. Featurise each molecule into a PyG molecular graph
  3. Load the tuned AttentiveFP checkpoint
  4. Predict P(EGFR active) for every compound
  5. Rank & filter to generate a prioritised lead list
  6. Compute ADMET-proxy properties (Lipinski, TPSA, QED)
  7. Save results: ranked CSV + top-hits visualisations

Usage:
    python run_virtual_screening.py
    python run_virtual_screening.py --top_n 50 --threshold 0.8
    python run_virtual_screening.py --skip_fetch   # reuse cached library
"""

import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
os.makedirs("results/screening", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from torch_geometric.loader import DataLoader as PyGDataLoader

from src.data.featurize import mol_to_graph
from src.models.attentivefp import AttentiveFPClassifier
from src.data.fetch_screening_library import fetch_library, SAVE_PATH

# ────────────────────────────────────────────────────────────
#  Best hyperparameters (from Optuna tuning)
# ────────────────────────────────────────────────────────────
BEST_HP = dict(
    in_channels=29, hidden_channels=256, out_channels=1,
    edge_dim=12, num_layers=3, num_timesteps=3, dropout=0.3067
)
CHECKPOINT = "checkpoints/ATTENTIVEFP_tuned_best.pt"


# ════════════════════════════════════════════════════════════
# 1. Model helpers
# ════════════════════════════════════════════════════════════

def load_model(device):
    model = AttentiveFPClassifier(**BEST_HP)
    if not os.path.exists(CHECKPOINT):
        sys.exit(
            f"[ERROR] Checkpoint not found at {CHECKPOINT}.\n"
            "Please run:  python run_interpretability.py  to train and save the model first."
        )
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=False))
    model.to(device).eval()
    print(f"Loaded model weights from {CHECKPOINT}")
    return model


# ════════════════════════════════════════════════════════════
# 2. Featurise screening library
# ════════════════════════════════════════════════════════════

def featurise_library(df: pd.DataFrame):
    """Convert every SMILES to a PyG Data object. Returns (graphs, valid_indices)."""
    graphs, valid_idx = [], []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Featurising", ncols=80):
        smi = row["canonical_smiles"]
        g = mol_to_graph(smi, label=0)        # label placeholder
        if g is not None:
            graphs.append(g)
            valid_idx.append(i)
    return graphs, valid_idx


# ════════════════════════════════════════════════════════════
# 3. Batch inference
# ════════════════════════════════════════════════════════════

def predict(graphs, model, device, batch_size=128):
    """Run inference in mini-batches; returns numpy array of P(active)."""
    loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", ncols=80):
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs  = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
            all_probs.extend(probs.tolist())
    return np.array(all_probs)


# ════════════════════════════════════════════════════════════
# 4. ADMET-proxy descriptors
# ════════════════════════════════════════════════════════════

def compute_properties(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "MW":           round(Descriptors.ExactMolWt(mol), 2),
        "LogP":         round(Descriptors.MolLogP(mol), 2),
        "HBD":          Descriptors.NumHDonors(mol),
        "HBA":          Descriptors.NumHAcceptors(mol),
        "TPSA":         round(Descriptors.TPSA(mol), 2),
        "RotBonds":     Descriptors.NumRotatableBonds(mol),
        "AromaticRings":Descriptors.NumAromaticRings(mol),
        "QED":          round(QED.qed(mol), 3),   # Drug-likeness score 0-1
        "Lipinski_Pass":int(
            Descriptors.ExactMolWt(mol) <= 500 and
            Descriptors.MolLogP(mol)    <= 5   and
            Descriptors.NumHDonors(mol) <= 5   and
            Descriptors.NumHAcceptors(mol) <= 10
        ),
    }


# ════════════════════════════════════════════════════════════
# 5. Visualise top-N hits
# ════════════════════════════════════════════════════════════

def visualise_top_hits(df_hits: pd.DataFrame, top_n: int = 20):
    """
    Grid of 2-D structures for the top-N predicted leads.
    Each cell shows ChEMBL ID, P(active), and QED.
    """
    cols = 4
    rows = int(np.ceil(top_n / cols))
    fig  = plt.figure(figsize=(cols * 4.5, rows * 4.5))
    fig.patch.set_facecolor("#0f1117")
    gs   = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.45, wspace=0.25)

    slice_ = df_hits.head(top_n)
    for k, (_, row) in enumerate(slice_.iterrows()):
        ax  = fig.add_subplot(gs[k // cols, k % cols])
        ax.set_facecolor("#1a1d2e")
        smi = row["canonical_smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            ax.axis("off")
            continue
        # render molecule to image
        try:
            drawer = rdMolDraw2D.MolDraw2DSVG(400, 320)
            drawer.drawOptions().padding = 0.1
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            import io
            from PIL import Image
            try:
                import cairosvg
                png = cairosvg.svg2png(bytestring=drawer.GetDrawingText().encode())
                img = Image.open(io.BytesIO(png))
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, smi[:40], ha="center", va="center",
                        color="white", fontsize=5, wrap=True, transform=ax.transAxes)
        except Exception:
            ax.text(0.5, 0.5, "render error", ha="center", va="center",
                    color="white", fontsize=8, transform=ax.transAxes)

        rank   = k + 1
        cid    = row.get("chembl_id", "unknown")
        prob   = row["p_active"]
        qed    = row.get("QED", "?")
        source = row.get("source_target", "?")
        ax.set_title(
            f"#{rank}  {cid}\nP(active)={prob:.3f}  QED={qed}\n[{source}]",
            color="white", fontsize=8, pad=4, fontweight="bold"
        )
        ax.axis("off")

    # hide unused subplots
    for k in range(len(slice_), rows * cols):
        ax = fig.add_subplot(gs[k // cols, k % cols])
        ax.axis("off")

    fig.suptitle(
        f"Virtual Screening: Top-{top_n} Predicted EGFR Inhibitors\n"
        f"(AttentiveFP · Tuned · Scaffold-Split Test AUROC 0.8415)",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )
    out = f"results/screening/top{top_n}_hits.png"
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved hit grid → {out}")
    return out


def visualise_score_distribution(probs: np.ndarray):
    """Histogram of predicted P(active) across the entire library."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")

    n, bins, patches = ax.hist(probs, bins=50, edgecolor="#ffffff22", linewidth=0.4)
    # colour by probability: low=grey, mid=blue, high=gold
    cmap = plt.cm.plasma
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(cmap(float(left)))

    ax.axvline(0.5,  color="#ff4d4d",  lw=1.5, linestyle="--", label="p=0.5 (decision)")
    ax.axvline(0.8,  color="#ffd700",  lw=1.5, linestyle="--", label="p=0.8 (high conf)")
    ax.axvline(0.9,  color="#00ff99",  lw=1.5, linestyle="--", label="p=0.9 (top hits)")

    n_hits = int((probs >= 0.5).sum())
    n_high = int((probs >= 0.8).sum())
    ax.text(0.72, 0.82,
            f"≥0.5 : {n_hits} compounds\n≥0.8 : {n_high} compounds",
            transform=ax.transAxes, color="white", fontsize=10,
            bbox=dict(facecolor="#0f1117", edgecolor="#444", boxstyle="round"))

    ax.set_xlabel("Predicted P(EGFR Active)", color="white", fontsize=12)
    ax.set_ylabel("Number of Compounds", color="white", fontsize=12)
    ax.set_title("Virtual Screening Score Distribution\n(AttentiveFP on unseen kinase-inhibitor library)",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=9)

    plt.tight_layout()
    out = "results/screening/score_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved distribution → {out}")
    return out


def visualise_property_scatter(df_hits: pd.DataFrame):
    """MW vs LogP scatter, point colour = P(active), size = QED."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")

    sc = ax.scatter(
        df_hits["LogP"], df_hits["MW"],
        c=df_hits["p_active"], cmap="plasma",
        s=df_hits["QED"].fillna(0.5) * 150 + 20,
        alpha=0.75, edgecolors="#ffffff22", linewidths=0.3,
        vmin=0, vmax=1
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("P(EGFR Active)", color="white", fontsize=11)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    # Lipinski Ro5 box
    ax.axvline(5,   color="#ff4d4d", lw=1, linestyle="--", alpha=0.6, label="LogP=5 (Ro5)")
    ax.axhline(500, color="#ffd700", lw=1, linestyle="--", alpha=0.6, label="MW=500 Da (Ro5)")

    ax.set_xlabel("LogP", color="white", fontsize=12)
    ax.set_ylabel("Molecular Weight (Da)", color="white", fontsize=12)
    ax.set_title("Chemical Space of Virtual Screening Hits\n(point size ∝ QED drug-likeness)",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=9)

    plt.tight_layout()
    out = "results/screening/chemical_space.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved chemical space → {out}")
    return out


# ════════════════════════════════════════════════════════════
# 6. Main
# ════════════════════════════════════════════════════════════

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Step 1: Get screening library ──────────────────────────────────────
    if args.skip_fetch and os.path.exists(SAVE_PATH):
        print(f"Loading cached screening library from {SAVE_PATH}")
        df_lib = pd.read_csv(SAVE_PATH)
    else:
        print("Fetching screening library from ChEMBL …")
        df_lib = fetch_library()

    print(f"Library size: {len(df_lib)} compounds\n")

    # ── Step 2: Featurise ──────────────────────────────────────────────────
    graphs, valid_idx = featurise_library(df_lib)
    df_valid = df_lib.loc[valid_idx].reset_index(drop=True)
    print(f"\nSuccessfully featurised: {len(graphs)} / {len(df_lib)} compounds")

    # ── Step 3: Load model & predict ──────────────────────────────────────
    model = load_model(device)
    probs = predict(graphs, model, device, batch_size=args.batch_size)

    df_valid["p_active"] = probs
    df_valid["rank"]     = df_valid["p_active"].rank(ascending=False).astype(int)

    # ── Step 4: Compute ADMET-proxy properties ────────────────────────────
    print("\nComputing ADMET-proxy properties …")
    props_list = []
    for smi in tqdm(df_valid["canonical_smiles"], desc="Properties", ncols=80):
        props_list.append(compute_properties(smi))
    props_df = pd.DataFrame(props_list)
    # Merge (avoid duplicate columns)
    for col in props_df.columns:
        df_valid[col] = props_df[col].values

    # ── Step 5: Filter & rank ──────────────────────────────────────────────
    df_hits = df_valid[df_valid["p_active"] >= args.threshold].copy()
    df_hits = df_hits.sort_values("p_active", ascending=False).reset_index(drop=True)
    df_hits["rank"] = range(1, len(df_hits) + 1)

    print(f"\n{'='*60}")
    print(f"VIRTUAL SCREENING SUMMARY")
    print(f"{'='*60}")
    print(f"Total compounds screened : {len(df_valid)}")
    print(f"Predicted active (≥{args.threshold}) : {len(df_hits)}")
    print(f"Predicted active (≥0.9)  : {int((probs >= 0.9).sum())}")
    print(f"Predicted active (≥0.95) : {int((probs >= 0.95).sum())}")
    print(f"\nTop 20 Lead Candidates:")
    print(f"{'Rank':<5} {'ChEMBL ID':<16} {'P(active)':<11} {'MW':<8} "
          f"{'LogP':<7} {'QED':<7} {'Ro5':<5} {'Source Target'}")
    print("-" * 80)
    for _, row in df_hits.head(20).iterrows():
        print(f"{int(row['rank']):<5} {str(row.get('chembl_id','?')):<16} "
              f"{row['p_active']:.4f}    "
              f"{row.get('MW','?'):<8} {row.get('LogP','?'):<7} "
              f"{row.get('QED','?'):<7} {row.get('Lipinski_Pass','?'):<5} "
              f"{row.get('source_target','?')}")
    print(f"{'='*60}\n")

    # ── Step 6: Save results ───────────────────────────────────────────────
    out_csv = "results/screening/ranked_leads.csv"
    df_hits.to_csv(out_csv, index=False)
    print(f"Full ranked list saved → {out_csv}")

    # Save top-50 separately for easy inspection
    df_hits.head(50).to_csv("results/screening/top50_leads.csv", index=False)

    # ── Step 7: Visualisations ─────────────────────────────────────────────
    print("\nGenerating visualisations …")
    visualise_score_distribution(probs)
    visualise_property_scatter(df_hits.head(200))
    if len(df_hits) > 0:
        visualise_top_hits(df_hits, top_n=min(args.top_n, len(df_hits)))

    print("\n✅ Virtual screening complete!")
    print("  Results:")
    print(f"    Ranked CSV  : results/screening/ranked_leads.csv")
    print(f"    Top-50 CSV  : results/screening/top50_leads.csv")
    print(f"    Figures     : results/screening/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual screening with tuned AttentiveFP")
    parser.add_argument("--skip_fetch",  action="store_true",
                        help="Skip ChEMBL fetch and use cached library")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Minimum P(active) to include in ranked list (default: 0.5)")
    parser.add_argument("--top_n",       type=int,   default=20,
                        help="Number of top hits to visualise in grid (default: 20)")
    parser.add_argument("--batch_size",  type=int,   default=128,
                        help="Inference batch size (default: 128)")
    args = parser.parse_args()
    main(args)
