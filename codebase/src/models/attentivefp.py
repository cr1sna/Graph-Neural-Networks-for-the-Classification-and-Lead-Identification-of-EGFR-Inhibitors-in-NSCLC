import torch
from torch_geometric.nn.models import AttentiveFP

class AttentiveFPClassifier(torch.nn.Module):
    def __init__(self, in_channels=29, hidden_channels=128, out_channels=1, edge_dim=12, num_layers=3, num_timesteps=2, dropout=0.2):
        super(AttentiveFPClassifier, self).__init__()
        
        # We wrap the PyG AttentiveFP model, which natively supports edge features
        self.model = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # AttentiveFP returns the raw output (logits)
        out = self.model(x, edge_index, edge_attr, batch)
        return out
