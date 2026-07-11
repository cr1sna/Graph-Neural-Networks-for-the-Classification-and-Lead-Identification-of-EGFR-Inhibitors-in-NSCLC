import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

class GINClassifier(torch.nn.Module):
    def __init__(self, node_dim=29, hidden_dim=128, num_layers=3, dropout=0.2, pooling='add'):
        super(GINClassifier, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        # GIN requires an MLP for the aggregation function
        for i in range(num_layers):
            in_dim = node_dim if i == 0 else hidden_dim
            
            mlp = Sequential(
                Linear(in_dim, hidden_dim), 
                ReLU(), 
                BatchNorm1d(hidden_dim), 
                Linear(hidden_dim, hidden_dim), 
                ReLU()
            )
            
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(BatchNorm1d(hidden_dim))
            
        # Pooling method
        if pooling == 'add':
            self.pool = global_add_pool
        elif pooling == 'mean':
            self.pool = global_mean_pool
        else:
            self.pool = global_add_pool
            
        self.fc1 = Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = Linear(hidden_dim // 2, 1)

    def forward(self, x, edge_index, batch):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        x = self.pool(x, batch)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        out = self.fc2(x)
        return out
