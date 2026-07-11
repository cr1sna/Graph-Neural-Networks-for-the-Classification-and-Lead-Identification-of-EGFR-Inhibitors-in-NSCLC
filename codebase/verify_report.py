"""verify_report.py — Consistency guard between code outputs and the thesis report.

Reads the canonical result files produced by the pipeline and checks that every
headline number appears in the LaTeX report, and that no known-stale value
remains anywhere. Run from the project root:

    python verify_report.py

Exits 0 if all checks pass, 1 otherwise.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

PROJECT = pathlib.Path(__file__).resolve().parent
RESULTS = PROJECT / "results"
REPORT = pathlib.Path("/Users/krishnagaire/Desktop/Reports/report_thesis")
TEX_DIRS = [REPORT / "chapters", REPORT / "front_pages"]

# Concatenate every .tex source into one searchable blob.
ALL_TEX = ""
TEX_FILES: list[pathlib.Path] = []
for d in TEX_DIRS:
    for f in sorted(d.glob("*.tex")):
        TEX_FILES.append(f)
        ALL_TEX += f.read_text()

checks: list[tuple[bool, str]] = []


def present(value: str, label: str) -> None:
    checks.append((value in ALL_TEX, f"PRESENT  {label}: '{value}'"))


def absent(value: str, label: str) -> None:
    checks.append((value not in ALL_TEX, f"ABSENT   {label}: '{value}'"))


# ---------------------------------------------------------------- load outputs
bench = pd.read_csv(RESULTS / "benchmark_results.csv").set_index("Model")
opt = json.load(open(RESULTS / "optuna_verify.json"))
# Three screens: AttentiveFP (comparison), and the two best models (GCN, RF).
leads_af = pd.read_csv(RESULTS / "nepali_leads.csv")
leads_gcn = pd.read_csv(RESULTS / "nepali_leads_gcn.csv")
leads_rf = pd.read_csv(RESULTS / "nepali_leads_rf.csv")
eda = json.load(open(RESULTS / "eda_summary.json"))


def _funnel(df):
    """[library, >=0.5, >=0.8, >=0.9, drug-like(>=0.9,Ro5,QED>=0.4)]."""
    p, ro5, qed = df["P_active"], df["ro5_compliant"], df["qed"]
    return [len(df), int((p >= .5).sum()), int((p >= .8).sum()),
            int((p >= .9).sum()), int(((p >= .9) & ro5 & (qed >= .4)).sum())]


# primary model for lead identity = Random Forest (best overall classifier)
leads = leads_rf
p = leads["P_active"]
ro5 = leads["ro5_compliant"]
qed = leads["qed"]

# ---------------------------------------------------------------- DATASET / EDA
present(f"{eda['n_compounds']:,}", "dataset size")
present(f"{eda['n_active']:,}", "active count")
present(f"{eda['n_inactive']:,}", "inactive count")
present(f"{eda['n_unique_scaffolds']:,}", "unique scaffolds")
present(f"{eda['n_singleton_scaffolds']:,}", "singleton scaffolds")
present(str(eda["largest_scaffold_size"]), "largest scaffold size")

# ---------------------------------------------------------------- SPLIT (80/10/10)
n = eda["n_compounds"]
present("8{,}418", "train size")
present("1{,}052", "val size")
present("1{,}053", "test size")

# ---------------------------------------------------------------- BENCHMARK
for model in bench.index:
    for metric in ["AUROC", "AUPRC", "F1", "MCC"]:
        present(f"{bench.loc[model, metric]:.4f}", f"{model} {metric}")

# ---------------------------------------------------------------- TUNING
present(f"{opt['best_val_auroc']:.4f}", "tuned best val AUROC")
bp = opt["best_params"]
present(f"{bp['dropout']:.3f}", "tuned dropout")          # 0.129
# Learning rate is typeset in the report as "$2.61 \times 10^{-3}$";
# check the mantissa is present rather than a fixed e-notation string.
present(f"{bp['lr']*1e3:.2f}", "tuned lr mantissa (x1e-3)")  # 2.61

# ---------------------------------------------------------------- SCREENING
present(f"{len(leads)}", "library size (535)")
# funnel counts for each of the three models must appear in tab:vs_summary
_af, _gcn, _rf = _funnel(leads_af), _funnel(leads_gcn), _funnel(leads_rf)
for stage, v in zip(["lib", ">=.5", ">=.8", ">=.9", "drug-like"], _af):
    present(str(v), f"AttentiveFP funnel {stage}")
for stage, v in zip(["lib", ">=.5", ">=.8", ">=.9", "drug-like"], _gcn):
    present(str(v), f"GCN funnel {stage}")
for stage, v in zip(["lib", ">=.5", ">=.8", ">=.9", "drug-like"], _rf):
    present(str(v), f"RF funnel {stage}")
# RF hit-rate percentages quoted in the text/appendix
present("33.8", "RF hit rate >=0.5 (%)")
present("4.5", "GCN hit rate >=0.5 (%)")

# per-species library counts (from the full RF screen of all 535 compounds)
for sp, cnt in leads.groupby("plant_species").size().items():
    present(str(int(cnt)), f"library count {sp}")

# top compliant lead under the best classifier (RF) = apigenin
top = leads[ro5].sort_values("P_active", ascending=False).iloc[0]
present(str(top["compound_name"]).title(), "top compliant lead name (Apigenin)")
present(f"{top['P_active']:.4f}", "top lead P_active (0.9127)")
# top consensus lead (active under BOTH best models) = luteolin
present("Luteolin", "consensus lead name")
present("luteolin", "consensus lead name lc")

# ---------------------------------------------------------------- PHYSCHEM TABLE
# Canonical physicochemical medians (recomputed and fixed this session).
present("478.0", "MW median (tab:physchem)")
present("96.3", "TPSA median (tab:physchem)")
present("47\\%", "Lipinski-compliant fraction")

# ---------------------------------------------------------------- FLOW DIAGRAM
present("Exploratory data analysis", "EDA node in flow diagram / methodology")
present("benchmark\\_results.csv across 7 models", "benchmark node in flow diagram")
present("29-dim node features + 12-dim edge features", "featurisation node")
present("Bemis-Murcko scaffold split (80/10/10)", "split node")

# ---------------------------------------------------------------- FIGURE DATA
# Guard against figure/table drift: the hard-coded data in the figure generator
# must match the output files exactly.
import importlib.util
import matplotlib
matplotlib.use("Agg")
_spec = importlib.util.spec_from_file_location(
    "mtf", PROJECT / "src" / "utils" / "make_thesis_figures.py")
_mtf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mtf)

_name_map = {"Random Forest": "RandomForest"}
_fig_ok = True
for i, mod in enumerate(_mtf.MODELS):
    csvname = _name_map.get(mod, mod)
    for metric in ["AUROC", "AUPRC", "F1", "MCC"]:
        if abs(_mtf.METRICS[metric][i] - round(float(bench.loc[csvname, metric]), 4)) > 5e-4:
            _fig_ok = False
checks.append((_fig_ok, "FIGURE   model-comparison data matches benchmark_results.csv"))

# funnel figure now compares all three models; each must match its CSV funnel
checks.append((_mtf.FUNNEL_MODELS["AttentiveFP (0.81)"] == _funnel(leads_af),
               "FIGURE   funnel AttentiveFP column matches nepali_leads.csv"))
checks.append((_mtf.FUNNEL_MODELS["GCN (0.85)"] == _funnel(leads_gcn),
               "FIGURE   funnel GCN column matches nepali_leads_gcn.csv"))
checks.append((_mtf.FUNNEL_MODELS["Random Forest (0.88)"] == _funnel(leads_rf),
               "FIGURE   funnel Random Forest column matches nepali_leads_rf.csv"))

_lib_fig = dict(_mtf.LIBRARY)
_lib_real = {k: int(v) for k, v in leads.groupby("plant_species").size().items()}
checks.append((_lib_fig == _lib_real, "FIGURE   library-composition data matches nepali_leads_rf.csv"))

_top_fig = _mtf.LEADS[0][0]
checks.append((_top_fig == str(top["compound_name"]).title(),
               "FIGURE   leads-scatter top compound matches nepali_leads_rf.csv (Apigenin)"))

# ---------------------------------------------------------------- STALE VALUES
for v, lab in [
    ("0.9973", "old tuned AUROC"), ("0.8936", "old RF AUROC"),
    ("0.8415", "old AttentiveFP AUROC"), ("0.9423", "old default val AUROC"),
    ("0.9641", "old AUPRC"), ("0.5656", "old MCC"),
    ("0.9821", "old berberine P"),
    ("0.9841", "old atranorin P (AttentiveFP top lead)"),
    ("atranorin", "old top lead atranorin"),
    ("Atranorin", "old top lead Atranorin"),
    ("26.9", "old AttentiveFP hit rate"),
    ("429.5", "old MW median"), ("3.41", "old LogP median"),
    ("1{,}872", "old kinase library"), ("CHEMBL150606", "old top hit"),
    ("100 Optuna trials", "old trial count"), ("100 trials", "old trial count 2"),
    ("MPS", "stale GPU backend"), ("Conda", "stale env"),
    ("[X]", "placeholder"),
]:
    absent(v, lab)

# ---------------------------------------------------------------- report
fails = [d for ok, d in checks if not ok]
passes = [d for ok, d in checks if ok]
print(f"\n{'='*64}\n REPORT CONSISTENCY VERIFICATION\n{'='*64}")
print(f" Checked {len(TEX_FILES)} .tex files against code outputs.")
print(f" PASS: {len(passes)}   FAIL: {len(fails)}")
if fails:
    print("\n FAILURES:")
    for d in fails:
        print(f"   [FAIL] {d}")
else:
    print("\n ALL CHECKS PASSED — report is consistent with code outputs.")
print("=" * 64)
sys.exit(1 if fails else 0)
