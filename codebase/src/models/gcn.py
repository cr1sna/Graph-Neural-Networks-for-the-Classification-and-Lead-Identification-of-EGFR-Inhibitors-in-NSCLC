import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

class GCNClassifier(torch.nn.Module):
    def __init__(self, node_dim=29, hidden_dim=128, num_layers=3, dropout=0.2, pooling='mean'):
        super(GCNClassifier, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GCN layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(node_dim, hidden_dim))
        self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
            
        # Pooling method
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        elif pooling == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Invalid pooling method: {pooling}")
            
        # Fully connected layers for classification
        self.fc1 = Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = Linear(hidden_dim // 2, 1)

    def forward(self, x, edge_index, batch):
        # Apply GNN layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Global pooling (graph-level readout)
        x = self.pool(x, batch)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output logit (will apply sigmoid/BCEWithLogitsLoss later)
        out = self.fc2(x)
        
        return out
