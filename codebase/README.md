# GNN-Based EGFR Inhibitor Activity Classification & Virtual Screening

> **Master's Thesis Project**  
> Binary classification of EGFR inhibitor activity using Graph Neural Networks,
> benchmarked against traditional ML baselines, with GNNExplainer interpretability
> and an end-to-end virtual screening pipeline for lead compound prioritisation.

---

## 🎯 Research Objectives

| # | Objective | Status |
|---|---|---|
| 1 | Develop & train GNN models for binary EGFR activity classification | ✅ Complete |
| 2 | Benchmark GNNs against traditional ML baselines (Random Forest, Logistic Regression) | ✅ Complete |
| 3 | Apply the best GNN model for virtual screening to identify and rank novel lead compounds | ✅ Complete |

---

## 📋 Project Overview

This project trains four state-of-the-art Graph Neural Network architectures to classify
molecular compounds as **EGFR inhibitors (active)** or **non-inhibitors (inactive)** based
on their graph-structured molecular representation.

**Data**: ~10,523 curated molecules from [ChEMBL203](https://www.ebi.ac.uk/chembl/target_report_card/CHEMBL203/)
(EGFR kinase), labelled using a pIC₅₀ ≥ 6.0 threshold.

**Splitting**: Strict [Bemis-Murcko Scaffold Split](https://pubs.acs.org/doi/10.1021/jm9602928)
(80% train / 10% val / 10% test) to evaluate generalisation to novel chemical scaffolds.

---

## 🏗️ Project Structure

```
.
├── run_pipeline.py             # Train & evaluate all GNN models + baselines
├── run_tuning.py               # Optuna hyperparameter optimisation
├── run_interpretability.py     # GNNExplainer analysis on known EGFR drugs
├── run_virtual_screening.py    # Virtual screening of unseen compound library
├── requirements.txt            # Pinned Python dependencies
│
├── src/
│   ├── data/
│   │   ├── fetch_chembl.py             # ChEMBL API data acquisition
│   │   ├── preprocess.py               # Molecule cleaning & labelling
│   │   ├── featurize.py                # SMILES → PyG molecular graph (29 node / 12 edge features)
│   │   ├── dataset.py                  # PyTorch Geometric MoleculeDataset
│   │   └── fetch_screening_library.py  # Fetch unseen compounds for virtual screening
│   │
│   ├── models/
│   │   ├── gcn.py              # Graph Convolutional Network
│   │   ├── gat.py              # Graph Attention Network
│   │   ├── gin.py              # Graph Isomorphism Network
│   │   ├── attentivefp.py      # AttentiveFP (best performer)
│   │   └── baselines.py        # Random Forest & Logistic Regression (ECFP4 fingerprints)
│   │
│   ├── training/
│   │   ├── trainer.py          # Training loop (early stopping, LR scheduling, weighted BCE)
│   │   ├── evaluate.py         # AUROC, AUPRC, F1, MCC, Accuracy metrics
│   │   └── tune.py             # Optuna objective + study runner
│   │
│   └── utils/
│       ├── splitting.py            # RDKit Murcko scaffold splitter
│       └── interpretability.py     # GNNExplainer wrapper & visualisation
│
├── data/
│   ├── raw/                    # Raw ChEMBL CSV downloads
│   ├── processed/              # Cleaned, labelled molecule CSV
│   ├── graphs/                 # PyG serialised graph objects (train/val/test)
│   └── screening_library.csv   # Virtual screening compound library
│
├── checkpoints/
│   └── ATTENTIVEFP_tuned_best.pt   # Best model weights (Val AUROC 0.9973)
│
├── results/
│   ├── tables/
│   │   └── final_comparison.csv    # Full 6-model benchmark table
│   ├── figures/                    # GNNExplainer visualisations for 4 known drugs
│   └── screening/
│       ├── ranked_leads.csv        # All predicted actives, ranked by P(active)
│       ├── top50_leads.csv         # Top-50 lead candidates with ADMET properties
│       ├── score_distribution.png  # Histogram of screening scores
│       ├── chemical_space.png      # LogP vs MW scatter coloured by P(active)
│       └── top20_hits.png          # 2D structure grid of top-20 leads
│
├── experiments/
│   └── logs/                   # TensorBoard training logs
│
└── configs/                    # YAML configuration files
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+ (tested on 3.14)
- macOS / Linux  
- **Note for macOS**: XGBoost requires `brew install libomp`. If unavailable, the pipeline uses Logistic Regression as the second baseline.

### Setup

```bash
# 1. Clone / navigate to the project
cd "Master's thesis"

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # macOS / Linux

# 3. Install PyTorch (CPU) — adjust for CUDA if available
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu

# 4. Install PyTorch Geometric
pip install torch-geometric==2.7.0

# 5. Install remaining dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Full pipeline (recommended)
```bash
# Runs everything end-to-end: ChEMBL fetch → preprocess → EDA → graph build →
# Bemis-Murcko scaffold split → train 7 models (5 GNNs + RF + MLP) → Optuna
# tuning (AttentiveFP) → Nepali medicinal-plant virtual screening.
# Writes results/: benchmark_results.csv, nepali_leads_rf.csv,
# nepali_leads_gcn.csv, optuna_verify.json, eda_summary.json.
python main.py
```

### Individual stages
```bash
python -m src.data.fetch_chembl        # 1. fetch EGFR bioactivity from ChEMBL
python -m src.data.preprocess          # 2. curate, pIC50, binary labels
python run_pipeline.py --run_baselines --epochs 200   # 3. train + benchmark 7 models
python run_tuning.py --model ATTENTIVEFP --n_trials 20 # 4. Optuna tuning (AttentiveFP)
python -m src.screening.screen_multimodel             # 5. GCN + Random Forest Nepali screen
```

### Reproduce figures & verify against the thesis
```bash
# Regenerate all thesis figures from the outputs
python -m src.utils.make_thesis_figures

# Consistency guard: checks the thesis .tex values against the code outputs.
# NOTE: results/ is not committed (regenerable), so run `python main.py` FIRST
# to generate the CSV/JSON outputs the guard reads — otherwise it will error
# on missing files.
python verify_report.py
```

---

## 📊 Results

### Model Comparison (Scaffold Test Set)

| Model | Type | AUROC | AUPRC | F1 | MCC | Accuracy |
|---|---|---|---|---|---|---|
| **Random Forest** | ML + ECFP4 | **0.8936** | **0.9641** | **0.9202** | **0.5656** | **0.8680** |
| **AttentiveFP (default)** | GNN | 0.8415 | 0.9498 | 0.8981 | 0.4796 | 0.8357 |
| **GAT** | GNN | 0.8192 | 0.9382 | 0.8747 | 0.4500 | 0.8063 |
| **Logistic Regression** | ML + ECFP4 | 0.8149 | 0.9310 | 0.8865 | 0.4794 | 0.8224 |
| **GCN** | GNN | 0.7995 | 0.9257 | 0.8951 | 0.4537 | 0.8300 |
| **GIN** | GNN | 0.7951 | 0.9244 | 0.8756 | 0.4178 | 0.8044 |

### After Optuna Hyperparameter Tuning

| Model | Val AUROC |
|---|---|
| AttentiveFP (default) | 0.9423 |
| **AttentiveFP (tuned)** | **0.9973** |

Best hyperparameters found:

| Parameter | Value |
|---|---|
| hidden_channels | 256 |
| num_layers | 3 |
| num_timesteps | 3 |
| dropout | 0.307 |
| batch_size | 64 |
| learning_rate | 4.33 × 10⁻⁴ |
| weight_decay | 8.95 × 10⁻⁴ |

### Virtual Screening

| Metric | Value |
|---|---|
| Compounds screened | 1,872 |
| Predicted active (P ≥ 0.5) | 878 (46.9%) |
| Predicted active (P ≥ 0.9) | 562 |
| **Top lead — CHEMBL150606** | P=1.000, MW=414 Da, LogP=3.85, QED=0.491, Lipinski ✅ |

---

## 🧬 Molecular Graph Representation

Each molecule is converted to a graph where:

**Node features (29 dimensions per atom):**
- Atom type one-hot: C, N, O, S, F, Cl, Br, I, P, Unknown (10 dims)
- Degree (6 dims)
- Formal charge (1 dim)
- Number of hydrogens (5 dims)
- Hybridisation: SP, SP2, SP3, SP3D, SP3D2 (5 dims)
- Aromaticity (1 dim)
- In ring (1 dim)

**Edge features (12 dimensions per bond):**
- Bond type: SINGLE, DOUBLE, TRIPLE, AROMATIC (4 dims)
- Conjugated (1 dim)
- In ring (1 dim)
- Stereo: STEREONONE, STEREOANY, STEREOZ, STEREOE, STEREOCIS, STEREOTRANS (6 dims)

---

## 🧠 Model Architectures

### GCN — Graph Convolutional Network
Stacked `GCNConv` layers with batch normalisation and global mean/max/add pooling.
Uses node features only.

### GAT — Graph Attention Network
Multi-head `GATv2Conv` layers that learn atom-level attention weights. Supports
edge features natively.

### GIN — Graph Isomorphism Network
`GINEConv` layers with MLP aggregators. Maximally expressive (Weisfeiler-Leman
bound). Supports edge features.

### AttentiveFP *(best performer)*
PyTorch Geometric's `AttentiveFP` model with multi-step graph-level attention
(timesteps) and gated recurrent units for message passing. Handles both node
and edge features natively.

---

## 🔍 Interpretability

GNNExplainer was applied to four approved EGFR kinase inhibitors to identify
which atoms and bonds drive the model's predictions:

| Drug | Generation | P(active) |
|---|---|---|
| Gefitinib (Iressa) | 1st gen | 0.9998 |
| Erlotinib (Tarceva) | 1st gen | 0.9995 |
| Lapatinib (Tykerb) | Dual EGFR/HER2 | 0.9998 |
| Osimertinib (Tagrisso) | 3rd gen (covalent) | 0.7179 |

**Key finding**: The GNN independently rediscovered the EGFR pharmacophore —
highlighting quinazoline ring nitrogens (hinge-binding), halogen substituents
(hydrophobic pocket), and aromatic core bonds without any explicit chemical
knowledge encoded.

---

## 📦 Dependencies

| Package | Version |
|---|---|
| torch | 2.12.0 |
| torch-geometric | 2.7.0 |
| rdkit | 2026.3.2 |
| scikit-learn | 1.8.0 |
| optuna | 4.8.0 |
| pandas | 3.0.3 |
| numpy | 2.4.6 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 |
| networkx | 3.6.1 |
| chembl-webresource-client | 0.10.9 |
| tqdm | 4.67.3 |
| tensorboard | 2.20.0 |
| Pillow | latest |
| requests | 2.34.2 |

---

## 📁 Key Output Files

| File | Description |
|---|---|
| `checkpoints/ATTENTIVEFP_tuned_best.pt` | Saved weights of the best model |
| `results/tables/final_comparison.csv` | 6-model benchmark metrics table |
| `results/screening/ranked_leads.csv` | All virtual screening hits, ranked |
| `results/screening/top50_leads.csv` | Top-50 lead candidates with ADMET properties |
| `results/figures/*_explanation.png` | GNNExplainer figures for known drugs |
| `results/screening/top20_hits.png` | Top-20 lead compound structure grid |

---

## 📚 References

1. Duvenaud et al. (2015). *Convolutional Networks on Graphs for Learning Molecular Fingerprints*. NeurIPS.
2. Veličković et al. (2018). *Graph Attention Networks*. ICLR.
3. Xu et al. (2019). *How Powerful are Graph Neural Networks?* ICLR.
4. Xiong et al. (2020). *Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism*. J. Med. Chem.
5. Bemis & Murcko (1996). *The Properties of Known Drugs. 1. Molecular Frameworks*. J. Med. Chem.
6. Ying et al. (2019). *GNNExplainer: Generating Explanations for Graph Neural Networks*. NeurIPS.
7. ChEMBL Database — EGFR (CHEMBL203): https://www.ebi.ac.uk/chembl/

---

## 🎓 Academic Use

This project was developed as part of a Master's thesis on graph-based deep learning
for drug discovery. All data is sourced from publicly available ChEMBL under the
[Creative Commons Attribution-ShareAlike 3.0 Unported License](https://creativecommons.org/licenses/by-sa/3.0/).
