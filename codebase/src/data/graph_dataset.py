"""graph_dataset.py — Convert cleaned SMILES to PyTorch Geometric Data objects.

Node features  : 29-dimensional per atom
Edge features  : 12-dimensional per bond
Label          : binary (1 = active, pIC50 >= 6.0; 0 = inactive)
"""

from __future__ import annotations

import logging
import pathlib
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdchem
from torch_geometric.data import Data, InMemoryDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature dimension constants (used as assertions throughout)
# ---------------------------------------------------------------------------
NODE_DIM: int = 29
EDGE_DIM: int = 12

# ---------------------------------------------------------------------------
# Atom-level feature helpers
# ---------------------------------------------------------------------------

ATOM_TYPES: list[str] = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]   # 9 known + 1 unknown = 10
HYBRIDISATION_TYPES: list = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]
STEREO_TYPES: list = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOCIS,
    rdchem.BondStereo.STEREOTRANS,
]


def _one_hot(value, choices: list, allow_unknown: bool = True) -> list[int]:
    """Return a one-hot list; last entry used for unknown if allow_unknown."""
    enc = [0] * (len(choices) + (1 if allow_unknown else 0))
    try:
        enc[choices.index(value)] = 1
    except ValueError:
        if allow_unknown:
            enc[-1] = 1
    return enc


def atom_features(atom: rdchem.Atom) -> list[int]:
    """Extract a 29-dimensional feature vector for a single atom.

    Dimensions:
        0-9  : Atom type one-hot (C,N,O,S,F,Cl,Br,I,P,Unknown)         — 10
        10-15: Degree one-hot (0,1,2,3,4,>=5)                           —  6
        16   : Formal charge (integer)                                   —  1
        17-21: Hydrogen count one-hot (0,1,2,3,>=4)                     —  5
        22-26: Hybridisation one-hot (SP,SP2,SP3,SP3D,SP3D2)            —  5
        27   : Is aromatic (binary)                                      —  1
        28   : Is in ring (binary)                                       —  1
                                                               Total  =  29

    Args:
        atom: RDKit atom object.

    Returns:
        List of integers of length NODE_DIM (29).
    """
    # Atom type (10 dim)
    feat = _one_hot(atom.GetSymbol(), ATOM_TYPES, allow_unknown=True)

    # Degree (6 dim): clamp >= 5 to last bucket
    degree = min(atom.GetDegree(), 5)
    feat += _one_hot(degree, [0, 1, 2, 3, 4, 5], allow_unknown=False)

    # Formal charge (1 dim)
    feat += [atom.GetFormalCharge()]

    # Hydrogen count (5 dim): clamp >= 4 to last bucket
    h_count = min(atom.GetTotalNumHs(), 4)
    feat += _one_hot(h_count, [0, 1, 2, 3, 4], allow_unknown=False)

    # Hybridisation (5 dim)
    feat += _one_hot(atom.GetHybridization(), HYBRIDISATION_TYPES, allow_unknown=False)

    # Aromaticity (1 dim)
    feat += [int(atom.GetIsAromatic())]

    # Ring membership (1 dim)
    feat += [int(atom.IsInRing())]

    assert len(feat) == NODE_DIM, (
        f"Expected node feature dim={NODE_DIM}, got {len(feat)}"
    )
    return feat


def bond_features(bond: rdchem.Bond) -> list[int]:
    """Extract a 12-dimensional feature vector for a single bond.

    Dimensions:
        0-3 : Bond type one-hot (SINGLE,DOUBLE,TRIPLE,AROMATIC)         —  4
        4   : Is conjugated (binary)                                     —  1
        5   : Is in ring (binary)                                        —  1
        6-11: Stereo one-hot (NONE,ANY,Z,E,CIS,TRANS)                   —  6
                                                               Total  =  12

    Args:
        bond: RDKit bond object.

    Returns:
        List of integers of length EDGE_DIM (12).
    """
    bond_type_map = {
        rdchem.BondType.SINGLE: 0,
        rdchem.BondType.DOUBLE: 1,
        rdchem.BondType.TRIPLE: 2,
        rdchem.BondType.AROMATIC: 3,
    }
    bt_enc = [0, 0, 0, 0]
    bt_enc[bond_type_map.get(bond.GetBondType(), 0)] = 1

    feat = bt_enc
    feat += [int(bond.GetIsConjugated())]
    feat += [int(bond.IsInRing())]
    feat += _one_hot(bond.GetStereo(), STEREO_TYPES, allow_unknown=False)

    assert len(feat) == EDGE_DIM, (
        f"Expected edge feature dim={EDGE_DIM}, got {len(feat)}"
    )
    return feat


# ---------------------------------------------------------------------------
# Single-molecule converter
# ---------------------------------------------------------------------------

def smiles_to_data(
    smiles: str,
    label: int,
    pic50: float,
) -> Optional[Data]:
    """Convert a SMILES string to a :class:`torch_geometric.data.Data` object.

    Args:
        smiles: Canonical SMILES string.
        label: Binary activity label (0 or 1).
        pic50: pIC50 value.

    Returns:
        A PyG ``Data`` object, or ``None`` if the SMILES is invalid or the
        molecule has no atoms after parsing.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            logger.warning("Cannot parse SMILES (skipping): %s", smiles)
            return None

        # Node features
        node_feats = [atom_features(a) for a in mol.GetAtoms()]
        x = torch.tensor(node_feats, dtype=torch.float)  # [N, 29]

        # Edge index + edge features (both directions)
        edge_indices: list[list[int]] = [[], []]
        edge_attrs: list[list[int]] = []

        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf = bond_features(bond)
            edge_indices[0] += [i, j]
            edge_indices[1] += [j, i]
            edge_attrs += [bf, bf]

        if edge_attrs:
            edge_index = torch.tensor(edge_indices, dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        else:
            # Single-atom molecule — no bonds
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, EDGE_DIM), dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.float),
            smiles=smiles,
            pIC50=torch.tensor([pic50], dtype=torch.float),
        )
        return data

    except Exception as exc:  # noqa: BLE001
        logger.warning("Error converting SMILES '%s': %s", smiles, exc)
        return None


# ---------------------------------------------------------------------------
# InMemoryDataset
# ---------------------------------------------------------------------------

class MoleculeDataset(InMemoryDataset):
    """PyG InMemoryDataset wrapping a cleaned ChEMBL EGFR CSV.

    Args:
        root: Root directory for raw/processed data cache.
        csv_path: Path to the cleaned CSV (columns: chembl_id, smiles,
            pIC50, label).
        transform: Optional PyG transform applied on-the-fly.
        pre_transform: Optional PyG transform applied at processing time.
    """

    def __init__(
        self,
        root: pathlib.Path | str,
        csv_path: pathlib.Path | str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        self.csv_path = pathlib.Path(csv_path)
        super().__init__(str(root), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0],
                                            weights_only=False)

    @property
    def raw_file_names(self) -> list[str]:
        return [self.csv_path.name]

    @property
    def processed_file_names(self) -> list[str]:
        return ["molecule_dataset.pt"]

    def download(self) -> None:
        pass  # Data already present via fetch_chembl + preprocess pipeline

    def process(self) -> None:
        """Build PyG Data objects from the cleaned CSV and cache to disk."""
        logger.info("Processing dataset from %s …", self.csv_path)
        df = pd.read_csv(self.csv_path)

        required = {"smiles", "pIC50", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}. "
                f"Columns found: {list(df.columns)}"
            )

        data_list: List[Data] = []
        skipped = 0

        for _, row in df.iterrows():
            data = smiles_to_data(
                smiles=str(row["smiles"]),
                label=int(row["label"]),
                pic50=float(row["pIC50"]),
            )
            if data is not None:
                data_list.append(data)
            else:
                skipped += 1

        logger.info(
            "Converted %d molecules (%d skipped) to PyG Data objects.",
            len(data_list), skipped,
        )

        if not data_list:
            raise ValueError("No valid molecules found — dataset is empty.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logger.info("Dataset saved to %s", self.processed_paths[0])
