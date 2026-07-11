import optuna
import torch
from torch_geometric.loader import DataLoader
from src.models.gcn import GCNClassifier
from src.models.gat import GATClassifier
from src.models.gin import GINClassifier
from src.models.attentivefp import AttentiveFPClassifier
from src.training.trainer import GNNTrainer
import copy

def get_model(model_name, trial, node_dim=29, edge_dim=12):
    """Dynamically creates a model based on trial suggestions."""
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    pooling = trial.suggest_categorical('pooling', ['mean', 'add', 'max'])
    
    if model_name == 'GCN':
        return GCNClassifier(node_dim=node_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, pooling=pooling)
    elif model_name == 'GAT':
        heads = trial.suggest_categorical('heads', [2, 4, 8])
        return GATClassifier(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, heads=heads, pooling=pooling)
    elif model_name == 'GIN':
        # GIN typically uses 'add' pooling, but we can search
        return GINClassifier(node_dim=node_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, pooling=pooling)
    elif model_name == 'ATTENTIVEFP':
        num_timesteps = trial.suggest_int('num_timesteps', 1, 3)
        return AttentiveFPClassifier(in_channels=node_dim, hidden_channels=hidden_dim, edge_dim=edge_dim, num_layers=num_layers, num_timesteps=num_timesteps, dropout=dropout)
    else:
        raise ValueError(f"Unknown model {model_name}")

def create_objective(model_name, train_dataset, val_dataset, device, pos_weight=None):
    """Creates the objective function for Optuna."""
    
    def objective(trial):
        # Data loaders
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Model
        model = get_model(model_name, trial)
        
        # Training config
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        
        config = {
            'lr': lr,
            'weight_decay': weight_decay,
            'lr_patience': 10,
            'patience': 20, # Early stopping patience
            'pos_weight': pos_weight
        }
        
        # Trainer
        log_dir = f"experiments/logs/optuna/{model_name}_trial_{trial.number}"
        trainer = GNNTrainer(model, device, config, log_dir=log_dir)
        
        # Train
        epochs = 100 # Max epochs for tuning
        best_val_auroc = -float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            trainer.train_epoch(train_loader)
            val_metrics = trainer.evaluate(val_loader)
            
            val_auroc = val_metrics['auroc']
            
            # Update learning rate
            trainer.scheduler.step(val_auroc)
            
            # Report to Optuna
            trial.report(val_auroc, epoch)
            
            # Handle pruning
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= config['patience']:
                break
                
        return best_val_auroc
        
    return objective

def run_tuning(model_name, train_dataset, val_dataset, device, n_trials=50, pos_weight=None):
    print(f"Starting hyperparameter tuning for {model_name}...")
    
    study = optuna.create_study(direction='maximize', study_name=f"{model_name}_tuning")
    objective = create_objective(model_name, train_dataset, val_dataset, device, pos_weight)
    
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial:")
    trial = study.best_trial
    print(f"  Value (Val AUROC): {trial.value}")
    print(f"  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    return study
