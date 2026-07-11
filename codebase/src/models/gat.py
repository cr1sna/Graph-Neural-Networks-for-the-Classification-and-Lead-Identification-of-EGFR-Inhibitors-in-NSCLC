import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool

class GATClassifier(torch.nn.Module):
    def __init__(self, node_dim=29, edge_dim=12, hidden_dim=128, num_layers=3, dropout=0.2, heads=4, pooling='mean'):
        super(GATClassifier, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GAT layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # Calculate dimension per head to keep total hidden_dim consistent
        dim_per_head = hidden_dim // heads
        
        # First layer (can use edge features)
        self.convs.append(GATConv(node_dim, dim_per_head, heads=heads, edge_dim=edge_dim, dropout=dropout))
        self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, dim_per_head, heads=heads, edge_dim=edge_dim, dropout=dropout))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
            
        # Pooling method
        if pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            self.pool = global_mean_pool
            
        # Fully connected layers
        self.fc1 = Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = Linear(hidden_dim // 2, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        # Apply GAT layers
        for i in range(self.num_layers):
            # GATConv takes edge_attr
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Global pooling
        x = self.pool(x, batch)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        out = self.fc2(x)
        return out
