"""baseline_models.py — Traditional ML baselines for EGFR classification.

Implements two baseline models trained on 2048-bit ECFP4 Morgan fingerprints:
    - RandomForestBaseline : Scikit-learn Random Forest (500 trees)
    - MLPBaseline          : 3-layer PyTorch MLP with BatchNorm + Dropout

Both expose a unified fit / predict_proba interface identical to the
sklearn convention, enabling drop-in comparison with GNN models.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FP_RADIUS: int = 2
FP_NBITS: int = 2048


# ---------------------------------------------------------------------------
# Fingerprint utility
# ---------------------------------------------------------------------------

def smiles_to_ecfp4(
    smiles_list: list[str],
    radius: int = FP_RADIUS,
    n_bits: int = FP_NBITS,
) -> np.ndarray:
    """Convert a list of SMILES strings to ECFP4 Morgan fingerprint matrix.

    Invalid or unparseable SMILES are replaced by zero vectors with a
    warning log entry.

    Args:
        smiles_list: List of canonical SMILES strings.
        radius: Morgan algorithm radius (default 2 → ECFP4).
        n_bits: Fingerprint bit length (default 2048).

    Returns:
        Float32 numpy array of shape ``(len(smiles_list), n_bits)``.
    """
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                logger.warning(
                    "Cannot parse SMILES (zero fingerprint): %s", smi
                )
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps[i] = np.array(fp, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fingerprint error for '%s': %s", smi, exc)
    return fps


# ---------------------------------------------------------------------------
# Random Forest Baseline
# ---------------------------------------------------------------------------

class RandomForestBaseline:
    """Random Forest classifier trained on ECFP4 Morgan fingerprints.

    Args:
        n_estimators: Number of decision trees (default 500).
        random_state: Random seed for reproducibility (default 42).
    """

    def __init__(
        self,
        n_estimators: int = 500,
        random_state: int = 42,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, smiles_list: list[str], labels: list[int]) -> None:
        """Train the Random Forest on ECFP4 fingerprints.

        Args:
            smiles_list: List of SMILES strings.
            labels: Binary integer labels (0 or 1).
        """
        logger.info("Computing ECFP4 fingerprints for %d molecules …",
                    len(smiles_list))
        X = smiles_to_ecfp4(smiles_list)
        y = np.array(labels, dtype=int)
        logger.info("Training Random Forest (%d trees) …", self.model.n_estimators)
        self.model.fit(X, y)
        self._fitted = True
        logger.info("Random Forest training complete.")

    def predict_proba(self, smiles_list: list[str]) -> np.ndarray:
        """Return predicted probability of the active class (label=1).

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            1-D float array of shape ``(len(smiles_list),)`` with P(active).

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba().")
        X = smiles_to_ecfp4(smiles_list)
        return self.model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# MLP Baseline (PyTorch)
# ---------------------------------------------------------------------------

class _MLPNet(nn.Module):
    """Internal 3-layer MLP: 2048 → 512 → 128 → 1."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FP_NBITS, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


class MLPBaseline:
    """3-layer MLP classifier trained on ECFP4 Morgan fingerprints.

    Uses BCEWithLogitsLoss with early stopping on validation loss.

    Args:
        hidden_dims: Not used directly (architecture is fixed for
            comparability with the thesis specification).
        dropout: Dropout probability (default 0.3).
        lr: Adam learning rate (default 1e-3).
        max_epochs: Maximum training epochs (default 100).
        patience: Early-stopping patience on validation loss (default 10).
        batch_size: Mini-batch size (default 256).
        random_state: Random seed (default 42).
    """

    def __init__(
        self,
        dropout: float = 0.3,
        lr: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 256,
        random_state: int = 42,
    ) -> None:
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state
        self._net: Optional[_MLPNet] = None
        self._fitted = False

        torch.manual_seed(random_state)

    def fit(self, smiles_list: list[str], labels: list[int]) -> None:
        """Train the MLP on ECFP4 fingerprints with early stopping.

        Uses 90 / 10 internal train/val split for early stopping.

        Args:
            smiles_list: List of SMILES strings.
            labels: Binary integer labels (0 or 1).
        """
        logger.info("Computing ECFP4 fingerprints for MLP (%d molecules) …",
                    len(smiles_list))
        X_np = smiles_to_ecfp4(smiles_list)
        y_np = np.array(labels, dtype=np.float32)

        # 90/10 internal split
        n = len(X_np)
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(n)
        split = int(0.9 * n)
        tr_idx, va_idx = idx[:split], idx[split:]

        X_tr = torch.tensor(X_np[tr_idx])
        y_tr = torch.tensor(y_np[tr_idx]).unsqueeze(1)
        X_va = torch.tensor(X_np[va_idx])
        y_va = torch.tensor(y_np[va_idx]).unsqueeze(1)

        # Positive class weight to handle imbalance
        pos_weight = torch.tensor(
            [(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
            dtype=torch.float,
        )

        self._net = _MLPNet(dropout=self.dropout)
        optimizer = optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_loss = float("inf")
        no_improve = 0
        best_state: dict = {}

        logger.info("Training MLP baseline …")
        self._net.train()

        for epoch in range(1, self.max_epochs + 1):
            # Mini-batch training
            perm = torch.randperm(len(X_tr))
            epoch_loss = 0.0
            for start in range(0, len(X_tr), self.batch_size):
                batch_idx = perm[start: start + self.batch_size]
                xb, yb = X_tr[batch_idx], y_tr[batch_idx]
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(xb)

            # Validation loss
            self._net.eval()
            with torch.no_grad():
                val_loss = criterion(self._net(X_va), y_va).item()
            self._net.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                logger.info("Early stopping at epoch %d (val_loss=%.4f)",
                            epoch, best_val_loss)
                break

        self._net.load_state_dict(best_state)
        self._net.eval()
        self._fitted = True
        logger.info("MLP training complete.")

    def predict_proba(self, smiles_list: list[str]) -> np.ndarray:
        """Return predicted probability of the active class (label=1).

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            1-D float array of shape ``(len(smiles_list),)`` with P(active).

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if not self._fitted or self._net is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X = torch.tensor(smiles_to_ecfp4(smiles_list))
        with torch.no_grad():
            logits = self._net(X).squeeze(1)
        return torch.sigmoid(logits).numpy()
