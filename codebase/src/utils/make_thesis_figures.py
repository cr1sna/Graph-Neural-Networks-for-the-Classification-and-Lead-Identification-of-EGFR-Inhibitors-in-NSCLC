"""make_thesis_figures.py — Generate all code-based figures for the thesis.

Produces two families of figures, written to the LaTeX report's ``figures/``
directory:

* **EDA figures** computed directly from the real cleaned EGFR dataset
  (``data/processed/egfr_cleaned.csv``): pIC50 distribution, class balance,
  physicochemical property distributions, and Bemis-Murcko scaffold diversity.
* **Result / screening figures** rendered from the benchmark and virtual
  screening values reported in the thesis, plus concept schematics
  (molecule-to-graph featurisation, GNN message passing).

All figures are saved as vector PDFs suitable for inclusion via
``\\includegraphics``. Run on CPU with a non-interactive backend.

Usage::

    python -m src.utils.make_thesis_figures
"""
from __future__ import annotations

import logging
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths and global style
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
CLEANED_CSV = PROJECT_ROOT / "data" / "processed" / "egfr_cleaned.csv"
FIG_DIR = pathlib.Path(
    "/Users/krishnagaire/Desktop/Reports/report_thesis/figures"
)

# Colour-blind-friendly palette
C_ACTIVE = "#2A7DB1"
C_INACTIVE = "#C44E52"
C_ACCENT = "#55A868"
C_GREY = "#7F7F7F"
PALETTE = ["#2A7DB1", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD", "#E07B39"]

plt.rcParams.update({
    # Times New Roman to match the thesis body text (IoST §10.6); fall back to
    # other Times variants then DejaVu if unavailable.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",      # Times-like math glyphs in figures
    "pdf.fonttype": 42,              # embed TrueType (not low-quality Type 3)
    "ps.fonttype": 42,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def _save(fig: plt.Figure, name: str) -> None:
    """Save a figure as PDF into the thesis figures directory."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    logger.info("Saved %s", out)


# ===========================================================================
# EDA figures — real EGFR dataset
# ===========================================================================

def fig_pic50_distribution(df: pd.DataFrame) -> None:
    """Histogram of pIC50 with the active/inactive decision threshold."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.linspace(df.pIC50.min(), min(df.pIC50.max(), 12), 60)
    ax.hist(df.loc[df.label == 1, "pIC50"], bins=bins, color=C_ACTIVE,
            alpha=0.85, label=f"Active (n={int((df.label==1).sum()):,})")
    ax.hist(df.loc[df.label == 0, "pIC50"], bins=bins, color=C_INACTIVE,
            alpha=0.85, label=f"Inactive (n={int((df.label==0).sum()):,})")
    ax.axvline(6.0, color="black", linestyle="--", linewidth=1.4)
    ax.text(6.05, ax.get_ylim()[1] * 0.92, "Activity threshold\n(pIC$_{50}$ = 6.0)",
            fontsize=9, va="top")
    ax.set_xlabel("pIC$_{50}$")
    ax.set_ylabel("Number of compounds")
    ax.set_title("Distribution of EGFR Bioactivity (pIC$_{50}$)")
    ax.legend(frameon=False)
    _save(fig, "eda_pic50_distribution.pdf")


def fig_class_balance(df: pd.DataFrame) -> None:
    """Donut chart of the active/inactive class balance."""
    n_act = int((df.label == 1).sum())
    n_ina = int((df.label == 0).sum())
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    wedges, _ = ax.pie(
        [n_act, n_ina], colors=[C_ACTIVE, C_INACTIVE],
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
    )
    ax.text(0, 0.12, f"{len(df):,}", ha="center", fontsize=15, fontweight="bold")
    ax.text(0, -0.16, "compounds", ha="center", fontsize=10, color=C_GREY)
    ax.legend(
        wedges,
        [f"Active  {100*n_act/len(df):.1f}%", f"Inactive  {100*n_ina/len(df):.1f}%"],
        loc="center", bbox_to_anchor=(0.5, -0.08), frameon=False, ncol=2, fontsize=10,
    )
    ax.set_title("EGFR Dataset Class Balance")
    _save(fig, "eda_class_balance.pdf")


def fig_property_distributions(df: pd.DataFrame, n_sample: int = 4000) -> None:
    """MW and LogP distributions with Lipinski reference lines."""
    sample = df.sample(min(n_sample, len(df)), random_state=0)
    mw, logp = [], []
    for smi in sample.smiles:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mw.append(Descriptors.MolWt(m))
            logp.append(Descriptors.MolLogP(m))
    mw, logp = np.array(mw), np.array(logp)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    axes[0].hist(mw, bins=50, color=C_ACCENT, alpha=0.9)
    axes[0].axvline(500, color="black", linestyle="--", linewidth=1.3)
    axes[0].text(505, axes[0].get_ylim()[1] * 0.9, "MW = 500 Da", fontsize=9, va="top")
    axes[0].set_xlabel("Molecular weight (Da)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Molecular Weight")

    axes[1].hist(logp, bins=50, color="#8172B3", alpha=0.9)
    axes[1].axvline(5, color="black", linestyle="--", linewidth=1.3)
    axes[1].text(5.1, axes[1].get_ylim()[1] * 0.9, "LogP = 5", fontsize=9, va="top")
    axes[1].set_xlabel("Calculated LogP")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Lipophilicity (LogP)")
    fig.suptitle(f"Physicochemical Property Distributions (n = {len(mw):,} sample)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "eda_property_distributions.pdf")


def fig_scaffold_diversity(df: pd.DataFrame, top_n: int = 12) -> None:
    """Top Bemis-Murcko scaffolds and the singleton fraction."""
    counts: dict[str, int] = {}
    for smi in df.smiles:
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            scaf = MurckoScaffold.GetScaffoldForMol(m)
            key = Chem.MolToSmiles(scaf, isomericSmiles=False)
            counts[key] = counts.get(key, 0) + 1
        except Exception:
            continue
    sizes = sorted(counts.values(), reverse=True)
    n_scaffolds = len(sizes)
    n_singletons = sum(1 for s in sizes if s == 1)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.7),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    top = sizes[:top_n]
    axes[0].barh(range(len(top))[::-1], top, color=C_ACTIVE)
    axes[0].set_yticks(range(len(top))[::-1])
    axes[0].set_yticklabels([f"Scaffold {i+1}" for i in range(len(top))], fontsize=9)
    axes[0].set_xlabel("Compounds sharing scaffold")
    axes[0].set_title(f"{top_n} Most Frequent Murcko Scaffolds")

    labels = ["Singleton\nscaffolds", "Multi-compound\nscaffolds"]
    vals = [n_singletons, n_scaffolds - n_singletons]
    axes[1].bar(labels, vals, color=[C_GREY, C_ACCENT])
    for i, v in enumerate(vals):
        axes[1].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("Number of scaffolds")
    axes[1].set_title(f"Scaffold Diversity\n({n_scaffolds:,} unique scaffolds)")
    fig.tight_layout()
    _save(fig, "eda_scaffold_diversity.pdf")


# ===========================================================================
# Concept figures
# ===========================================================================

def fig_molecule_to_graph() -> None:
    """Illustrate SMILES -> 2D structure -> attributed graph featurisation."""
    import networkx as nx

    smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"  # gefitinib
    mol = Chem.MolFromSmiles(smiles)

    fig = plt.figure(figsize=(9.2, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.1])

    # Panel A: RDKit 2D depiction
    axA = fig.add_subplot(gs[0, 0])
    img = Draw.MolToImage(mol, size=(420, 360))
    axA.imshow(img)
    axA.axis("off")
    axA.set_title("(a) Molecular structure (gefitinib)")

    # Panel B: attributed graph
    axB = fig.add_subplot(gs[0, 1])
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), sym=atom.GetSymbol(),
                   arom=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    pos = nx.kamada_kawai_layout(G)
    node_colors = [C_ACCENT if G.nodes[n]["arom"] else C_ACTIVE for n in G.nodes]
    nx.draw_networkx_edges(G, pos, ax=axB, edge_color=C_GREY, width=1.4)
    nx.draw_networkx_nodes(G, pos, ax=axB, node_color=node_colors,
                           node_size=210, edgecolors="white", linewidths=1.0)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["sym"] for n in G.nodes},
                            ax=axB, font_size=7, font_color="white", font_family="serif")
    axB.axis("off")
    axB.set_title("(b) Attributed molecular graph")
    axB.text(0.5, -0.06,
             "Each atom -> 29-dim node feature vector\n"
             "Each bond -> 12-dim edge feature vector",
             transform=axB.transAxes, ha="center", fontsize=9, color="black")
    fig.tight_layout()
    _save(fig, "molecule_to_graph.pdf")


def fig_message_passing() -> None:
    """Schematic of one GNN message-passing / neighbourhood aggregation step."""
    import networkx as nx

    G = nx.Graph()
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6)]
    G.add_edges_from(edges)
    pos = {0: (0, 0), 1: (-1.1, 0.9), 2: (1.1, 0.9), 3: (0, -1.25),
           4: (-2.0, 1.6), 5: (2.0, 1.6), 6: (0, -2.4)}

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#CCCCCC", width=1.5)
    neigh = [1, 2, 3]
    nx.draw_networkx_nodes(G, pos, nodelist=[0], node_color=C_INACTIVE,
                           node_size=900, ax=ax, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=neigh, node_color=C_ACTIVE,
                           node_size=700, ax=ax, edgecolors="white", linewidths=1.5)
    nx.draw_networkx_nodes(G, pos, nodelist=[4, 5, 6], node_color=C_GREY,
                           node_size=520, ax=ax, edgecolors="white", linewidths=1.5)
    # Aggregation arrows from neighbours to centre
    for n in neigh:
        arr = FancyArrowPatch(pos[n], pos[0], arrowstyle="-|>", mutation_scale=16,
                              color=C_ACCENT, lw=2.0, shrinkA=16, shrinkB=18)
        ax.add_patch(arr)
    labels = {0: "v", 1: "u$_1$", 2: "u$_2$", 3: "u$_3$", 4: "", 5: "", 6: ""}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=11,
                            font_color="white", font_weight="bold", font_family="serif")
    ax.text(0.5, -0.02, "Messages from neighbours $u_i$ are aggregated\n"
                        "to update the central atom embedding $h_v$",
            transform=ax.transAxes, ha="center", va="top", fontsize=10)
    ax.set_title("Graph Neural Network Message Passing")
    ax.axis("off")
    ax.margins(0.12)
    fig.subplots_adjust(bottom=0.16)
    _save(fig, "gnn_message_passing.pdf")


# ===========================================================================
# Result / screening figures — thesis-reported values
# ===========================================================================

# Benchmark values exactly as reported in chapter 4 (tab:results), ordered by AUROC.
MODELS = ["Random Forest", "MLP", "GCN", "GraphSAGE", "GIN", "GAT", "AttentiveFP"]
METRICS = {
    "AUROC":    [0.8827, 0.8547, 0.8454, 0.8229, 0.8217, 0.8162, 0.8087],
    "AUPRC":    [0.9496, 0.9354, 0.9332, 0.9179, 0.9163, 0.9176, 0.9108],
    "F1":       [0.8762, 0.8629, 0.7877, 0.8070, 0.8487, 0.7689, 0.8254],
    "MCC":      [0.5072, 0.5281, 0.4707, 0.4706, 0.4865, 0.4473, 0.3989],
}

# Virtual screening funnel (chapter 4, tab:vs_summary) — real COCONUT screen under
# the three models. Reveals the model-sensitivity of the hit list.
FUNNEL_STAGES = [
    "Full library",
    r"$\hat{p}\geq0.50$",
    r"$\hat{p}\geq0.80$",
    r"$\hat{p}\geq0.90$",
    "Drug-like\nleads",
]
FUNNEL_MODELS = {
    "AttentiveFP (0.81)":   [527, 143, 83, 71, 11],
    "GCN (0.85)":           [527,  24,  1,  0,  0],
    "Random Forest (0.88)": [527, 179,  4,  0,  0],
}

# Library composition (chapter 4, tab:library_comp) — COCONUT per-species counts
# after removing 8 compounds already present in the ChEMBL data (535 -> 527).
LIBRARY = [
    ("Nardostachys jatamansi", 199), ("Terminalia chebula", 192),
    ("Tinospora cordifolia", 71), ("Swertia chirayita", 23),
    ("Berberis aristata", 19), ("Rhododendron arboreum", 15),
    ("Ocimum sanctum", 8),
]

# Random Forest (best model) drug-like leads (chapter 4, tab:leads_rf), on the
# de-duplicated library: name, P_active, MW, LogP, QED, Ro5.
LEADS = [
    ("Isogentisin", 0.7282, 258.2, 2.37, 0.655, True),
    ("Luteolin", 0.6789, 286.2, 2.28, 0.511, True),
    ("Tinosporaside", 0.6607, 492.5, 0.63, 0.440, True),
    ("Columbin", 0.6180, 358.4, 2.53, 0.613, True),
    ("Palmarin", 0.6042, 374.4, 1.74, 0.591, True),
    ("Nardostachysin", 0.5836, 430.5, 2.95, 0.526, True),
]


def fig_model_comparison() -> None:
    """Grouped bar chart of the four headline metrics across all seven models."""
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    x = np.arange(len(MODELS))
    metric_names = list(METRICS.keys())
    width = 0.2
    for i, mname in enumerate(metric_names):
        ax.bar(x + (i - 1.5) * width, METRICS[mname], width,
               label=mname, color=PALETTE[i])
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance on the Scaffold-Split Test Set")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _save(fig, "results_model_comparison.pdf")


def fig_screening_funnel() -> None:
    """Grouped bars comparing the screening funnel across the three models."""
    stages = FUNNEL_STAGES
    models = list(FUNNEL_MODELS.keys())
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(stages))
    width = 0.26
    for i, m in enumerate(models):
        vals = FUNNEL_MODELS[m]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=m, color=PALETTE[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 4, str(v),
                    ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Compounds passing filter")
    ax.set_ylim(0, 560)
    ax.set_title("Nepali Plant Virtual Screening Funnel by Model")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _save(fig, "screening_funnel.pdf")


def fig_library_composition() -> None:
    """Bar chart of compounds retrieved per Nepali plant species."""
    names = [n for n, _ in LIBRARY]
    vals = [v for _, v in LIBRARY]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    bars = ax.bar(range(len(names)), vals, color=PALETTE[:len(names)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center",
                va="bottom", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8, style="italic")
    ax.set_ylabel("Number of compounds")
    ax.set_title("Nepali Medicinal Plant Screening Library Composition")
    _save(fig, "screening_library_composition.pdf")


def fig_leads_scatter() -> None:
    """P(active) vs QED scatter for the Random Forest drug-like leads (size = MW).

    Every listed lead is Lipinski-compliant, so no compliance legend is needed;
    marker area encodes molecular weight and the dashed lines mark the H2 thresholds.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, p, mw, logp, qed, ro5 in LEADS:
        ax.scatter(qed, p, s=mw, c=[C_ACTIVE], alpha=0.8,
                   edgecolors="black", linewidths=0.6, zorder=3)
        ax.annotate(name, (qed, p), xytext=(5, 4), textcoords="offset points",
                    fontsize=8)
    ax.axhline(0.80, color=C_GREY, linestyle="--", linewidth=1.1)
    ax.axvline(0.40, color=C_GREY, linestyle="--", linewidth=1.1)
    ax.set_xlim(0.36, 0.73)
    ax.set_ylim(0.55, 0.96)
    ax.text(0.435, 0.905,
            "Dashed lines: $H_2$ thresholds\n($\\hat{p}$ $\\geq$ 0.80, QED $\\geq$ 0.40)",
            fontsize=8, color=C_GREY, va="center", ha="left")
    ax.set_xlabel("QED (drug-likeness)")
    ax.set_ylabel(r"Predicted $\hat{p}$ (active)")
    ax.set_title("Random Forest Nepali Plant Leads: Activity vs Drug-likeness")
    ax.grid(linestyle=":", alpha=0.45)
    _save(fig, "screening_leads_scatter.pdf")


# Consensus of drug-like predicted-active leads (p>=0.5, Ro5, QED>=0.4) between the
# two best models on the de-duplicated library (chapter 4, fig:consensus).
CONSENSUS = {
    "rf_only": 22, "gcn_only": 1, "both": 3,
    "leads": [
        ("Luteolin",       "\\textit{T.~chebula}",    0.679, 0.514),
        ("Palmarin",       "\\textit{T.~cordifolia}", 0.604, 0.553),
        ("Nardostachysin", "\\textit{N.~jatamansi}",  0.584, 0.592),
    ],
}


def fig_consensus_venn() -> None:
    """Two-set Venn of drug-like predicted-active leads (Random Forest vs GCN),
    with the consensus (intersection) leads named."""
    from matplotlib.patches import Circle
    C_RF, C_GCN = C_ACTIVE, "#55A868"
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.add_patch(Circle((-0.5, 0), 1.5, facecolor=C_RF, alpha=0.28,
                        edgecolor=C_RF, lw=1.8))
    ax.add_patch(Circle((1.05, 0), 1.1, facecolor=C_GCN, alpha=0.30,
                        edgecolor=C_GCN, lw=1.8))
    # set titles
    ax.text(-1.55, 1.7, "Random Forest", color=C_RF, fontsize=12,
            fontweight="bold", ha="center")
    ax.text(1.65, 1.35, "GCN", color="#3d7a4e", fontsize=12,
            fontweight="bold", ha="center")
    # region counts
    ax.text(-1.15, 0.05, str(CONSENSUS["rf_only"]), fontsize=17, ha="center", va="center")
    ax.text(-1.15, -0.45, "RF-only\ndrug-like actives", fontsize=8, ha="center",
            va="top", color=C_GREY)
    ax.text(1.62, 0.05, str(CONSENSUS["gcn_only"]), fontsize=15, ha="center", va="center")
    ax.text(1.62, -0.4, "GCN-only", fontsize=8, ha="center", va="top", color=C_GREY)
    ax.text(0.32, 0.35, str(CONSENSUS["both"]), fontsize=18, fontweight="bold",
            ha="center", va="center", color="#222")
    ax.text(0.32, -0.2, "consensus", fontsize=8.5, ha="center", va="top", color="#222")
    # named consensus leads below the diagram
    leads_txt = "Consensus leads:  " + ";  ".join(
        f"{n} ({rf:.2f}/{gc:.2f})" for n, _, rf, gc in CONSENSUS["leads"])
    leads_txt += "\n(marker: Random Forest $\\hat{p}$ / GCN $\\hat{p}$)"
    ax.text(0.0, -2.15, leads_txt, fontsize=8.5, ha="center", va="top")
    ax.set_xlim(-3.0, 2.9); ax.set_ylim(-2.9, 2.1)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Drug-like predicted-active leads: cross-model consensus "
                 "(Random Forest $\\cap$ GCN)", fontsize=11.5)
    _save(fig, "screening_consensus_venn.pdf")


def main() -> None:
    """Generate the complete figure set."""
    if not CLEANED_CSV.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {CLEANED_CSV}. Run the preprocessing "
            "pipeline first."
        )
    df = pd.read_csv(CLEANED_CSV)
    logger.info("Loaded %d compounds for EDA figures.", len(df))

    # EDA (real data)
    fig_pic50_distribution(df)
    fig_class_balance(df)
    fig_property_distributions(df)
    fig_scaffold_diversity(df)

    # Concept schematics
    fig_molecule_to_graph()
    fig_message_passing()

    # Results / screening (thesis-reported values)
    fig_model_comparison()
    fig_screening_funnel()
    fig_library_composition()
    fig_leads_scatter()
    fig_consensus_venn()

    logger.info("All thesis figures generated in %s", FIG_DIR)


if __name__ == "__main__":
    main()
