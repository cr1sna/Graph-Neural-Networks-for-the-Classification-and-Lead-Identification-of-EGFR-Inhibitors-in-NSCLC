"""optuna_tuner.py — Bayesian hyperparameter optimisation for AttentiveFP."""
from __future__ import annotations
import csv, logging, pathlib
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch_geometric.loader import DataLoader
from src.models.gnn_models import AttentiveFPModel
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)
RESULTS_DIR = pathlib.Path("results")

def run_optuna(
    train_data, val_data,
    n_trials: int = 20,
    epochs: int = 100,
    seed: int = 42,
) -> tuple[dict, float]:
    """Bayesian hyperparameter search for AttentiveFP using Optuna TPE.

    Search space:
        hidden_channels : {64, 128, 256}
        num_layers      : [2, 4]
        num_timesteps   : [2, 4]
        dropout         : log-uniform [0.1, 0.5]
        batch_size      : {32, 64, 128}
        lr              : log-uniform [1e-4, 1e-2]
        weight_decay    : log-uniform [1e-5, 1e-3]

    Args:
        train_data: Training Data list.
        val_data: Validation Data list.
        n_trials: Number of Optuna trials (default 20).
        epochs: Training epochs per trial (default 100).
        seed: Random seed for sampler.

    Returns:
        Tuple of (best_params dict, best_val_auroc float).
    """
    trial_rows = []

    def objective(trial: optuna.Trial) -> float:
        hc  = trial.suggest_categorical("hidden_channels", [64, 128, 256])
        nl  = trial.suggest_int("num_layers", 2, 4)
        nt  = trial.suggest_int("num_timesteps", 2, 4)
        do  = trial.suggest_float("dropout", 0.1, 0.5, log=True)
        bs  = trial.suggest_categorical("batch_size", [32, 64, 128])
        lr  = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd  = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        model = AttentiveFPModel(
            hidden_channels=hc,
            num_layers=nl,
            num_timesteps=nt,
            dropout=do,
        )
        trainer = Trainer(
            lr=lr, weight_decay=wd, batch_size=bs, patience=15,
            log_file=RESULTS_DIR / f"trial_{trial.number}_log.csv",
        )
        metrics = trainer.train(model, train_data, val_data, epochs=epochs)
        auroc = metrics.get("AUROC", 0.0)

        trial_rows.append({
            "trial": trial.number,
            "hidden_channels": hc, "num_layers": nl, "num_timesteps": nt,
            "dropout": do, "batch_size": bs, "lr": lr, "weight_decay": wd,
            "val_auroc": auroc,
        })
        return auroc

    sampler = TPESampler(seed=seed)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials)

    # Save all trials
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trials_csv = RESULTS_DIR / "optuna_trials.csv"
    if trial_rows:
        keys = list(trial_rows[0].keys())
        with open(trials_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(trial_rows)
    logger.info("Optuna trials saved to %s", trials_csv)

    best = study.best_trial
    logger.info("Best trial #%d  AUROC=%.4f", best.number, best.value)
    logger.info("Best params: %s", best.params)
    return best.params, best.value
