import os
import argparse
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from src.data.dataset import MoleculeDataset
from src.utils.splitting import scaffold_split_data
from src.models.gcn import GCNClassifier
from src.models.gat import GATClassifier
from src.models.gin import GINClassifier
from src.models.attentivefp import AttentiveFPClassifier
from src.training.trainer import GNNTrainer
from src.models.baselines import prepare_baseline_data, train_evaluate_rf, train_evaluate_lr
from src.training.evaluate import print_metrics

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Check data
    processed_data_path = 'data/processed/egfr_cleaned.csv'
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Processed data not found at {processed_data_path}. Run preprocessing first.")
        
    df = pd.read_csv(processed_data_path)
    print(f"Loaded {len(df)} samples.")
    
    # 2. Split Data
    train_idx, val_idx, test_idx = scaffold_split_data(df)
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    # 3. Create PyG Datasets
    print("\nCreating Graph Datasets...")
    os.makedirs('data/graphs/raw', exist_ok=True)
    train_df.to_csv('data/graphs/raw/train.csv', index=False)
    val_df.to_csv('data/graphs/raw/val.csv', index=False)
    test_df.to_csv('data/graphs/raw/test.csv', index=False)
    
    train_dataset = MoleculeDataset(root='data/graphs', filename='train.csv', test=False)
    
    val_dataset = MoleculeDataset(root='data/graphs', filename='val.csv', test=False)
    
    test_dataset = MoleculeDataset(root='data/graphs', filename='test.csv', test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Class imbalance weight
    num_actives = train_df['label'].sum()
    num_inactives = len(train_df) - num_actives
    pos_weight = float(num_inactives / num_actives) if num_actives > 0 else 1.0
    print(f"Calculated pos_weight: {pos_weight:.2f}")
    
    results = {}
    
    # 4. Train Traditional ML Baselines
    if args.run_baselines:
        print("\n--- Running Traditional ML Baselines ---")
        X_train, y_train = prepare_baseline_data(train_df)
        X_val, y_val = prepare_baseline_data(val_df)
        X_test, y_test = prepare_baseline_data(test_df)
        
        _, rf_metrics = train_evaluate_rf(X_train, y_train, X_test, y_test)
        print("\nRandom Forest Test Metrics:")
        print_metrics(rf_metrics)
        results['Random Forest'] = rf_metrics
        
        _, lr_metrics = train_evaluate_lr(X_train, y_train, X_test, y_test)
        print("\nLogistic Regression Test Metrics:")
        print_metrics(lr_metrics)
        results['Logistic Regression'] = lr_metrics

    # 5. Train GNN Models
    models_to_run = args.models.split(',') if args.models else []
    
    for model_name in models_to_run:
        model_name = model_name.strip().upper()
        if not model_name: continue
            
        print(f"\n--- Training {model_name} ---")
        
        if model_name == 'GCN':
            model = GCNClassifier(node_dim=29, hidden_dim=128, num_layers=3, dropout=0.2)
        elif model_name == 'GAT':
            model = GATClassifier(node_dim=29, edge_dim=12, hidden_dim=128, num_layers=3, dropout=0.2)
        elif model_name == 'GIN':
            model = GINClassifier(node_dim=29, hidden_dim=128, num_layers=3, dropout=0.2)
        elif model_name == 'ATTENTIVEFP':
            model = AttentiveFPClassifier(in_channels=29, hidden_channels=128, edge_dim=12, num_layers=3)
        else:
            print(f"Unknown model: {model_name}. Skipping.")
            continue
            
        config = {
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'lr_patience': 10,
            'patience': 20,
            'pos_weight': pos_weight
        }
        
        trainer = GNNTrainer(model, device, config, log_dir=f"experiments/logs/{model_name}")
        trained_model = trainer.train(train_loader, val_loader, epochs=args.epochs, save_path=f"checkpoints/{model_name}_best.pt")
        
        print(f"\nEvaluating {model_name} on Test Set:")
        test_metrics = trainer.evaluate(test_loader)
        print_metrics(test_metrics)
        results[model_name] = test_metrics
        
    # 6. Save Results Summary
    if results:
        df_res = pd.DataFrame(results).T
        df_res.to_csv('results/tables/final_comparison.csv')
        print("\nResults saved to results/tables/final_comparison.csv")
        print(df_res[['auroc', 'auprc', 'f1', 'mcc', 'accuracy']])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN EGFR Classification Pipeline")
    parser.add_argument('--models', type=str, default='GCN,GAT', help='Comma-separated list of models to run (GCN, GAT, GIN, ATTENTIVEFP)')
    parser.add_argument('--run_baselines', action='store_true', help='Run RF and XGBoost baselines')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    
    args = parser.parse_args()
    main(args)
