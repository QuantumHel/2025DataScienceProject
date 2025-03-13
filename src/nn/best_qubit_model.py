import torch
import torch.nn as nn
import torch.nn.functional as F

class BestQubitModel(nn.Module):
    def __init__(self, n_size=4, hidden_layers=6, hidden_size=64, dropout_rate=0.5):
        super(BestQubitModel, self).__init__()
        input_size = 2 * n_size * n_size  # 32 features
        output_size = n_size * n_size     # 16 outputs

        self.n_size = n_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate

        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Hidden layers
        self.hidden_layers_list = nn.ModuleList()
        for _ in range(hidden_layers - 1):
            self.hidden_layers_list.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)
        
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer with adjustable dropout rat

    def forward(self, x):
        # Take only the first two channels: shape becomes [batch, 2, n_size, n_size]
        x = x[:, :2, :, :]
        # Flatten to [batch, 2*n_size*n_size]
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        
        for layer in self.hidden_layers_list:
            x = F.relu(layer(x))
            x = self.dropout(x)  # Apply dropout
        
        x = self.fc_out(x)
        
        # Reshape output to match target shape: [batch, 1, n_size, n_size]
        x = x.view(-1, 1, self.n_size, self.n_size)
        return x