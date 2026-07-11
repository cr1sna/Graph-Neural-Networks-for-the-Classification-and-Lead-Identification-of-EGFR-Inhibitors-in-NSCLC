import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import random

def generate_scaffold(smiles, include_chirality=False):
    """Compute the Bemis-Murcko scaffold for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def scaffold_split_data(df, smiles_col='smiles', frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=42):
    """
    Splits a pandas DataFrame into train, valid, and test sets using Murcko scaffolds.
    This ensures that molecules with similar core structures are grouped together,
    providing a more realistic measure of a model's ability to generalize to novel compounds.
    """
    print(f"Applying pure RDKit Scaffold Split (train: {frac_train}, valid: {frac_valid}, test: {frac_test})...")
    np.random.seed(seed)
    random.seed(seed)
    
    # 1. Group molecules by scaffold
    scaffolds = defaultdict(list)
    for i, smiles in enumerate(df[smiles_col]):
        scaffold = generate_scaffold(smiles)
        scaffolds[scaffold].append(i)
        
    # 2. Sort scaffolds by size (descending) to ensure large scaffold families don't accidentally all fall into a tiny test set
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]
    
    # 3. Distribute scaffold sets into train/val/test
    train_cutoff = frac_train * len(df)
    valid_cutoff = (frac_train + frac_valid) * len(df)
    
    train_idx, valid_idx, test_idx = [], [], []
    
    for scaffold_set in scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)
            
    # Check class distributions
    print("\nClass distribution after split:")
    for name, indices in zip(['Train', 'Valid', 'Test'], [train_idx, valid_idx, test_idx]):
        labels = df.loc[indices, 'label']
        actives = labels.sum()
        total = len(labels)
        print(f"{name}: {total} samples, Actives: {actives} ({actives/total:.1%} if total > 0 else 0%)")
        
    return train_idx, valid_idx, test_idx

if __name__ == "__main__":
    pass
