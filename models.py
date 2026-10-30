import torch
import torch.nn as nn

class DynamicGRUClassifier(nn.Module):
    """
    1-2 Layer GRU model for classifying landmark sequence buffers (T x 63) into whole-word signs.
    """
    def __init__(self, input_size=63, hidden_size=64, num_layers=2, num_classes=10):
        super(DynamicGRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.gru(x)
        # Take hidden state of last sequence step
        out = self.fc(out[:, -1, :])
        return out