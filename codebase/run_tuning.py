import os
import argparse
import pandas as pd
import torch
from src.data.dataset import MoleculeDataset
from src.training.tune import run_tuning

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nLoading Graph Datasets...")
    train_dataset = MoleculeDataset(root='data/graphs', filename='train.csv', test=False)
    val_dataset = MoleculeDataset(root='data/graphs', filename='val.csv', test=False)
    
    # Calculate pos_weight based on train dataset
    train_df = pd.read_csv('data/graphs/raw/train.csv')
    num_actives = train_df['label'].sum()
    num_inactives = len(train_df) - num_actives
    pos_weight = float(num_inactives / num_actives) if num_actives > 0 else 1.0
    print(f"Calculated pos_weight: {pos_weight:.2f}")
    
    study = run_tuning(
        model_name=args.model.upper(), 
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        device=device, 
        n_trials=args.n_trials, 
        pos_weight=pos_weight
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ATTENTIVEFP', help='Model to tune (e.g. ATTENTIVEFP, GAT)')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of optuna trials')
    
    args = parser.parse_args()
    main(args)
