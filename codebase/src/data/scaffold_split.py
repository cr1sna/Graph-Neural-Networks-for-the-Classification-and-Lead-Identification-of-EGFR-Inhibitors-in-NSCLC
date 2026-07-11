"""scaffold_split.py — Bemis-Murcko scaffold-aware dataset splitting.

Splits a list of PyG Data objects into train/val/test partitions such that
all molecules sharing the same Bemis-Murcko scaffold are confined to a
single partition. This prevents data leakage between sets and provides a
realistic evaluation of generalisation to novel chemical scaffolds.

Reference:
    Bemis, G.W.; Murcko, M.A. J. Med. Chem. 1996, 39, 2887-2893.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Optional

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scaffold helper
# ---------------------------------------------------------------------------

def _get_murcko_scaffold(smiles: str) -> Optional[str]:
    """Return the canonical Bemis-Murcko scaffold SMILES for a molecule.

    Args:
        smiles: Input canonical SMILES string.

    Returns:
        Canonical scaffold SMILES, or ``None`` if RDKit cannot parse the
        molecule or extract a scaffold.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, isomericSmiles=False, canonical=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Scaffold extraction failed for '%s': %s", smiles, exc)
        return None


# ---------------------------------------------------------------------------
# Public split function
# ---------------------------------------------------------------------------

def scaffold_split(
    dataset: list[Data],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> tuple[list[Data], list[Data], list[Data]]:
    """Split a dataset by Bemis-Murcko scaffold to avoid data leakage.

    Algorithm:
        1. Compute the Murcko scaffold for every molecule.
        2. Group molecule indices by scaffold SMILES.
        3. Sort scaffold groups largest-first (large groups → training set
           to preserve training set size).
        4. Assign scaffold groups greedily to train/val/test until each
           partition reaches its target fraction.
        5. Distribute singletons (molecules with a unique scaffold or no
           scaffold) proportionally.

    Args:
        dataset: List of :class:`torch_geometric.data.Data` objects, each
            having a ``smiles`` attribute.
        frac_train: Fraction of molecules for the training set (default 0.8).
        frac_val: Fraction for the validation set (default 0.1).
        frac_test: Fraction for the test set (default 0.1).
        seed: Random seed for reproducible singleton distribution.

    Returns:
        A ``(train_set, val_set, test_set)`` tuple of Data lists.

    Raises:
        ValueError: If fractions do not sum to 1.0 (within 1e-6 tolerance),
            or if ``dataset`` is empty.
    """
    if abs(frac_train + frac_val + frac_test - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1.0, got "
            f"{frac_train} + {frac_val} + {frac_test} = "
            f"{frac_train + frac_val + frac_test:.6f}"
        )
    if len(dataset) == 0:
        raise ValueError("dataset is empty.")

    rng = random.Random(seed)
    n = len(dataset)

    # ---- 1. Build scaffold → [indices] map --------------------------------
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    no_scaffold_indices: list[int] = []

    for idx, data in enumerate(dataset):
        smiles = getattr(data, "smiles", None)
        if smiles is None:
            no_scaffold_indices.append(idx)
            continue
        scaffold = _get_murcko_scaffold(smiles)
        if scaffold is None:
            no_scaffold_indices.append(idx)
        else:
            scaffold_to_indices[scaffold].append(idx)

    # ---- 2. Sort groups: largest first ------------------------------------
    scaffold_groups = sorted(
        scaffold_to_indices.values(), key=len, reverse=True
    )

    # Separate singletons (scaffold groups of size 1) for proportional dist.
    multi_groups = [g for g in scaffold_groups if len(g) > 1]
    singleton_groups = [g for g in scaffold_groups if len(g) == 1]
    singleton_indices = [g[0] for g in singleton_groups] + no_scaffold_indices

    # ---- 3. Greedy assignment of multi-compound scaffold groups -----------
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    train_target = int(frac_train * n)
    val_target = int(frac_val * n)

    for group in multi_groups:
        if len(train_idx) < train_target:
            train_idx.extend(group)
        elif len(val_idx) < val_target:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    # ---- 4. Fill remaining capacity with singletons to reach targets -----
    # Multi-compound scaffold groups are placed first (largest to train); the
    # singletons then top up each partition to its target size, which yields the
    # intended overall fractions even when large scaffolds dominate the train set.
    rng.shuffle(singleton_indices)
    for idx in singleton_indices:
        if len(train_idx) < train_target:
            train_idx.append(idx)
        elif len(val_idx) < val_target:
            val_idx.append(idx)
        else:
            test_idx.append(idx)

    # ---- 5. Build split datasets -----------------------------------------
    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]

    # ---- 6. Diversity report ---------------------------------------------
    n_scaffolds = len(scaffold_to_indices)
    n_singletons = len(singleton_groups)

    def _class_balance(split: list[Data]) -> str:
        labels = [int(d.y.item()) for d in split]
        if not labels:
            return "n/a"
        pct = 100 * sum(labels) / len(labels)
        return f"{pct:.1f}% active"

    logger.info("─── Scaffold Split Summary ───────────────────────────────")
    logger.info("Total molecules      : %d", n)
    logger.info("Unique scaffolds     : %d", n_scaffolds)
    logger.info("Singleton scaffolds  : %d  (%.1f%%)",
                n_singletons, 100 * n_singletons / max(n_scaffolds, 1))
    logger.info("No-scaffold molecules: %d", len(no_scaffold_indices))
    logger.info("Train  : %5d molecules  |  %s", len(train_set),
                _class_balance(train_set))
    logger.info("Val    : %5d molecules  |  %s", len(val_set),
                _class_balance(val_set))
    logger.info("Test   : %5d molecules  |  %s", len(test_set),
                _class_balance(test_set))
    logger.info("─────────────────────────────────────────────────────────")

    return train_set, val_set, test_set
