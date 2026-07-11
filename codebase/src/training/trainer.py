"""trainer.py — Unified training and evaluation loop for GNN models."""
from __future__ import annotations
import csv, logging, pathlib
import torch, torch.nn as nn, torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)
RESULTS_DIR = pathlib.Path("results")

class Trainer:
    """Train and evaluate a GNN on molecular graph datasets.

    Args:
        lr: Adam learning rate.
        weight_decay: L2 regularisation.
        batch_size: DataLoader batch size.
        patience: Early-stopping patience on val AUROC.
        scheduler_patience: ReduceLROnPlateau patience.
        scheduler_factor: LR reduction factor.
        log_file: Per-epoch CSV log path.
    """
    def __init__(self, lr=1e-3, weight_decay=0.0, batch_size=64,
                 patience=20, scheduler_patience=10, scheduler_factor=0.5,
                 log_file=RESULTS_DIR/"training_log.csv"):
        self.lr=lr; self.weight_decay=weight_decay; self.batch_size=batch_size
        self.patience=patience; self.scheduler_patience=scheduler_patience
        self.scheduler_factor=scheduler_factor
        self.log_file=pathlib.Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def train(self, model, train_data, val_data, epochs=200):
        """Train model; return best validation metrics dict."""
        train_loader=DataLoader(train_data,batch_size=self.batch_size,shuffle=True)
        val_loader=DataLoader(val_data,batch_size=self.batch_size,shuffle=False)
        labels=torch.cat([d.y for d in train_data])
        pos_weight=torch.tensor([(labels==0).sum().item()/max((labels==1).sum().item(),1)],dtype=torch.float)
        criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer=optim.Adam(model.parameters(),lr=self.lr,weight_decay=self.weight_decay)
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=self.scheduler_factor,patience=self.scheduler_patience)
        best_auroc=-1.0; best_metrics={}; best_state={}; no_improve=0
        with open(self.log_file,"w",newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_loss","val_auroc"])
        for epoch in range(1,epochs+1):
            tl=self._train_epoch(model,train_loader,optimizer,criterion)
            vl,vm=self._eval_epoch(model,val_loader,criterion)
            va=vm.get("AUROC",0.0); scheduler.step(va)
            with open(self.log_file,"a",newline="") as f:
                csv.writer(f).writerow([epoch,f"{tl:.4f}",f"{vl:.4f}",f"{va:.4f}"])
            if epoch%10==0 or epoch==1:
                logger.info("Epoch %3d | train=%.4f | val=%.4f | AUROC=%.4f",epoch,tl,vl,va)
            if va>best_auroc:
                best_auroc=va; best_metrics=vm
                best_state={k:v.clone() for k,v in model.state_dict().items()}
                no_improve=0
            else:
                no_improve+=1
            if no_improve>=self.patience:
                logger.info("Early stopping epoch %d (AUROC=%.4f).",epoch,best_auroc); break
        model.load_state_dict(best_state)
        logger.info("Training complete. Best AUROC=%.4f",best_auroc)
        return best_metrics

    def evaluate(self, model, data_list):
        """Return metrics dict for data_list."""
        loader=DataLoader(data_list,batch_size=self.batch_size,shuffle=False)
        _,metrics=self._eval_epoch(model,loader,nn.BCEWithLogitsLoss())
        return metrics

    def save_checkpoint(self, model, path):
        """Save model state dict to path (.pt)."""
        path=pathlib.Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),path); logger.info("Saved %s",path)

    def load_checkpoint(self, model, path):
        """Load model weights from path. Raises FileNotFoundError if missing."""
        path=pathlib.Path(path)
        if not path.exists(): raise FileNotFoundError(f"Checkpoint not found: {path}")
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        logger.info("Loaded %s",path)

    @staticmethod
    def _train_epoch(model,loader,optimizer,criterion):
        model.train(); tl=0.0; tn=0
        for b in loader:
            optimizer.zero_grad()
            loss=criterion(model(b.x,b.edge_index,b.edge_attr,b.batch).squeeze(1),b.y.float())
            loss.backward(); optimizer.step()
            tl+=loss.item()*b.num_graphs; tn+=b.num_graphs
        return tl/max(tn,1)

    @staticmethod
    def _eval_epoch(model,loader,criterion):
        model.eval(); logits_all=[]; labels_all=[]; tl=0.0; tn=0
        with torch.no_grad():
            for b in loader:
                lg=model(b.x,b.edge_index,b.edge_attr,b.batch).squeeze(1)
                tl+=criterion(lg,b.y.float()).item()*b.num_graphs; tn+=b.num_graphs
                logits_all.append(lg); labels_all.append(b.y)
        probs=torch.sigmoid(torch.cat(logits_all)).cpu().numpy()
        labels=torch.cat(labels_all).cpu().numpy()
        return tl/max(tn,1), compute_metrics(labels,probs)
