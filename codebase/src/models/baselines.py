import numpy as np
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import AllChem
from src.training.evaluate import compute_metrics

def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Calculates Morgan fingerprint from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def prepare_baseline_data(df, smiles_col='smiles', label_col='label'):
    """Converts a dataframe of SMILES to a numpy array of fingerprints and labels."""
    print("Calculating Morgan Fingerprints...")
    X = np.array([get_morgan_fingerprint(s) for s in df[smiles_col]])
    y = df[label_col].values
    return X, y

def train_evaluate_rf(X_train, y_train, X_val, y_val, n_estimators=100, max_depth=None, random_state=42):
    """Trains and evaluates a Random Forest classifier."""
    print(f"Training Random Forest (n_estimators={n_estimators})...")
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        n_jobs=-1, 
        random_state=random_state,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Get probabilities for class 1
    y_pred_probs = model.predict_proba(X_val)[:, 1]
    
    metrics = compute_metrics(y_val, y_pred_probs)
    return model, metrics

from sklearn.linear_model import LogisticRegression

def train_evaluate_lr(X_train, y_train, X_val, y_val, max_iter=1000, random_state=42):
    """Trains and evaluates a Logistic Regression classifier."""
    print(f"Training Logistic Regression...")
    
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Get probabilities for class 1
    y_pred_probs = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_pred_probs)
    return model, metrics
