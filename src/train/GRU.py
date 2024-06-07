import torch
import torch.nn as nn
import numpy as np

def mask_layer(shape, P):
    if not (0 <= P <= 1):
        raise ValueError("P must be between 0 and 1.")
    random_tensor = torch.rand(shape) 
    result_tensor = (random_tensor >= P).float()
    return result_tensor

#-------------------------------------------------------------------------------------------------

class GRUAmanda(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, mask_prob, dropout_prob, l1_lambda):
        super(GRUAmanda, self).__init__()
        self.norm      = nn.LayerNorm(input_size)
        self.mask_prob = mask_prob
        self.gru       = nn.GRU(input_size, hidden_size, batch_first=True)
        self.relu      = nn.ReLU() 
        self.fc        = nn.Linear(hidden_size, num_classes)
        self.dropout   = nn.Dropout(dropout_prob) 
        self.softmax   = nn.Softmax(dim=1)
        self.l1_lambda = l1_lambda 
 
    def forward(self, x):
        out    = self.norm(x) 
        out    = out * mask_layer(out.shape, self.mask_prob).to(out.device)
        out, _ = self.gru(out)
        out    = self.relu(out[:, -1, :])
        out    = self.dropout(out)
        out    = self.softmax(self.fc(out))

        return out
