"""main.py — End-to-end EGFR GNN pipeline orchestrator.

Runs the full pipeline:
  1. Fetch + preprocess EGFR ChEMBL data
  2. Exploratory data analysis of the cleaned dataset
  3. Build molecular graph dataset
  4. Bemis-Murcko scaffold split (80/10/10)
  5. Train + evaluate 7 models (GCN, GAT, GIN, AttentiveFP, GraphSAGE, RF, MLP)
  6. Optuna Bayesian tuning on AttentiveFP
  7. Nepali medicinal plant virtual screening
  8. Save benchmark_results.csv and nepali_leads.csv

Usage:
    python main.py
    python main.py --skip-tuning --skip-screening --epochs 50 --seed 0
"""
from __future__ import annotations
import argparse, csv, logging, pathlib, random
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("main")

RESULTS_DIR  = pathlib.Path("results")
DATA_RAW     = pathlib.Path("data/raw/egfr_chembl_raw.csv")
DATA_CLEAN   = pathlib.Path("data/processed/egfr_cleaned.csv")
DATASET_ROOT = pathlib.Path("data/pyg_cache")
CHECKPOINT   = RESULTS_DIR / "best_attentivefp.pt"


def set_seeds(seed: int) -> None:
    """Fix Python, NumPy, and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Random seeds fixed to %d.", seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EGFR GNN Pipeline")
    p.add_argument("--skip-tuning",    action="store_true", help="Skip Optuna tuning")
    p.add_argument("--skip-screening", action="store_true", help="Skip Nepali plant screening")
    p.add_argument("--epochs",  type=int, default=200, help="Max training epochs (default 200)")
    p.add_argument("--trials",  type=int, default=20,  help="Optuna trials (default 20)")
    p.add_argument("--seed",    type=int, default=42,  help="Random seed (default 42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # AttentiveFP architecture used for the saved screening checkpoint.
    # Defaults match AttentiveFPModel(); overwritten if Optuna tuning runs so
    # that screening reconstructs the exact architecture before loading weights.
    screening_model_kwargs: dict = {}

    # ── Step 1: Fetch raw data ──────────────────────────────────────────────
    if not DATA_RAW.exists():
        logger.info("Step 1: Fetching EGFR data from ChEMBL …")
        from src.data.fetch_chembl import fetch_egfr_data
        fetch_egfr_data()
    else:
        logger.info("Step 1: Raw data already exists at %s.", DATA_RAW)

    # ── Step 2: Preprocess ──────────────────────────────────────────────────
    if not DATA_CLEAN.exists():
        logger.info("Step 2: Preprocessing raw data …")
        from src.data.preprocess import preprocess_data
        preprocess_data()
    else:
        logger.info("Step 2: Cleaned data already exists at %s.", DATA_CLEAN)

    # ── Step 3: Exploratory data analysis (before graph construction) ───────
    logger.info("Step 3: Running exploratory data analysis …")
    from src.data.eda import run_eda
    run_eda(csv_path=DATA_CLEAN)

    # ── Step 4: Build PyG dataset ───────────────────────────────────────────
    logger.info("Step 4: Building molecular graph dataset …")
    from src.data.graph_dataset import MoleculeDataset
    dataset = MoleculeDataset(root=str(DATASET_ROOT), csv_path=DATA_CLEAN)
    data_list = [dataset[i] for i in range(len(dataset))]
    logger.info("Dataset size: %d molecules.", len(data_list))

    # ── Step 5: Scaffold split ──────────────────────────────────────────────
    logger.info("Step 5: Applying Bemis-Murcko scaffold split (80/10/10) …")
    from src.data.scaffold_split import scaffold_split
    train_data, val_data, test_data = scaffold_split(
        data_list, frac_train=0.8, frac_val=0.1, frac_test=0.1, seed=args.seed
    )

    # ── Step 6: Train & evaluate all 7 models ──────────────────────────────
    logger.info("Step 6: Training and evaluating 7 models …")
    from src.models.gnn_models import (
        GCNModel, GATModel, GINModel, AttentiveFPModel, GraphSAGEModel
    )
    from src.models.baseline_models import RandomForestBaseline, MLPBaseline
    from src.training.trainer import Trainer
    from src.utils.metrics import print_metrics_table

    all_results: dict[str, dict] = {}
    trainer = Trainer(batch_size=64, patience=20, log_file=RESULTS_DIR/"training_log.csv")

    gnn_models = {
        "GCN":         GCNModel(),
        "GAT":         GATModel(),
        "GIN":         GINModel(),
        "AttentiveFP": AttentiveFPModel(),
        "GraphSAGE":   GraphSAGEModel(),
    }

    for name, model in gnn_models.items():
        logger.info("  Training %s (%d params) …", name, model.get_num_params())
        trainer.log_file = RESULTS_DIR / f"{name}_log.csv"
        trainer.train(model, train_data, val_data, epochs=args.epochs)
        metrics = trainer.evaluate(model, test_data)
        all_results[name] = metrics
        logger.info("  %s test AUROC = %.4f", name, metrics.get("AUROC", 0))
        # Persist the default-config AttentiveFP so virtual screening has a
        # usable checkpoint even when Optuna tuning is skipped. If tuning runs,
        # this is overwritten by the tuned model below.
        if name == "AttentiveFP":
            trainer.save_checkpoint(model, CHECKPOINT)

    # Baselines
    train_smiles = [d.smiles for d in train_data]
    train_labels = [int(d.y.item()) for d in train_data]
    test_smiles  = [d.smiles for d in test_data]
    test_labels  = [int(d.y.item()) for d in test_data]

    from src.utils.metrics import compute_metrics
    for BClass, bname in [(RandomForestBaseline, "RandomForest"), (MLPBaseline, "MLP")]:
        logger.info("  Training %s …", bname)
        b = BClass()
        b.fit(train_smiles, train_labels)
        probs = b.predict_proba(test_smiles)
        all_results[bname] = compute_metrics(np.array(test_labels), probs)
        logger.info("  %s test AUROC = %.4f", bname, all_results[bname].get("AUROC", 0))

    # Save benchmark results
    bench_csv = RESULTS_DIR / "benchmark_results.csv"
    metrics_keys = ["AUROC","AUPRC","F1","MCC","Accuracy","Precision","Recall"]
    with open(bench_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Model"]+metrics_keys)
        w.writeheader()
        for mname, mdict in all_results.items():
            w.writerow({"Model": mname, **{k: round(mdict.get(k,0),4) for k in metrics_keys}})
    logger.info("Benchmark results saved to %s", bench_csv)
    print_metrics_table(all_results)

    # ── Step 7: Optuna tuning ───────────────────────────────────────────────
    if not args.skip_tuning:
        logger.info("Step 7: Running Optuna Bayesian tuning for AttentiveFP …")
        from src.training.optuna_tuner import run_optuna
        best_params, best_auroc = run_optuna(
            train_data, val_data, n_trials=args.trials, epochs=min(args.epochs, 100),
            seed=args.seed,
        )
        logger.info("Best params: %s  →  val AUROC %.4f", best_params, best_auroc)

        # Retrain with best params and save checkpoint
        screening_model_kwargs = {
            "hidden_channels": best_params["hidden_channels"],
            "num_layers": best_params["num_layers"],
            "num_timesteps": best_params["num_timesteps"],
            "dropout": best_params["dropout"],
        }
        tuned = AttentiveFPModel(**screening_model_kwargs)
        tuned_trainer = Trainer(
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"],
            batch_size=best_params["batch_size"],
            patience=30,
            log_file=RESULTS_DIR/"tuned_attentivefp_log.csv",
        )
        tuned_trainer.train(tuned, train_data, val_data, epochs=args.epochs)
        tuned_trainer.save_checkpoint(tuned, CHECKPOINT)
        tuned_metrics = tuned_trainer.evaluate(tuned, test_data)
        all_results["AttentiveFP (tuned)"] = tuned_metrics
        logger.info("Tuned AttentiveFP test AUROC = %.4f", tuned_metrics.get("AUROC", 0))
    else:
        logger.info("Step 7: Skipping Optuna tuning (--skip-tuning).")

    # ── Step 8: Nepali plant screening ──────────────────────────────────────
    if not args.skip_screening:
        logger.info("Step 8: Running Nepali medicinal plant virtual screening …")
        if not CHECKPOINT.exists():
            logger.warning(
                "Checkpoint %s not found. Run without --skip-tuning first, "
                "or place a trained model checkpoint at that path.", CHECKPOINT
            )
        else:
            from src.data.fetch_nepali_plants import fetch_nepali_plant_compounds
            plant_csv = pathlib.Path("data/nepali_plants/nepali_compounds.csv")
            if not plant_csv.exists():
                fetch_nepali_plant_compounds()
            from src.screening.screen_nepali import screen_nepali_compounds
            screen_nepali_compounds(
                checkpoint=CHECKPOINT,
                model_kwargs=screening_model_kwargs,
            )
    else:
        logger.info("Step 8: Skipping Nepali plant screening (--skip-screening).")

    logger.info("Pipeline complete. Results saved to %s/", RESULTS_DIR)


if __name__ == "__main__":
    main()
