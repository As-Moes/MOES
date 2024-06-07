import torch
import torch.nn as nn
import numpy as np

def mask_layer(shape, P):
    if not (0 <= P <= 1):
        raise ValueError("P must be between 0 and 1.")
    random_tensor = torch.rand(shape) 
    result_tensor = (random_tensor >= P).float()
    return result_tensor

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        attn_weights = torch.tanh(self.attn(encoder_outputs))
        attn_weights = torch.sum(attn_weights * self.v, dim=2)
        attn_weights = torch.softmax(attn_weights, dim=1)
        new_hidden = torch.sum(encoder_outputs * attn_weights.unsqueeze(2), dim=1)
        return new_hidden

class ImprovedLSTM(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size, mask_prob, dropout_prob, l1_lambda):
        super(ImprovedLSTM, self).__init__()
        self.norm = nn.LayerNorm(input_size)
        self.mask_prob = mask_prob
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_prob)
        self.softmax = nn.Softmax(dim=1)
        self.l1_lambda = l1_lambda

    def forward(self, x):
        out = self.norm(x)
        out = out * mask_layer(out.shape, self.mask_prob).to(out.device)
        out, _ = self.lstm(out)
        out = self.attention(out, out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.softmax(self.fc2(out))

        return out
