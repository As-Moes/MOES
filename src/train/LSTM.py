
import torch
import torch.nn as nn

class LSTMModelSimple(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModelSimple, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc   = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU() 
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out    = self.fc(out[:, -1, :])
        out    = self.softmax(out) 
        return out

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, fc1_size, fc2_size, num_classes, dropout_prob=0.5, l1_lambda=0.001):
        super(LSTMModel, self).__init__()
        self.norm = nn.LayerNorm(input_size)  # Apply normalization to the input
        self.dropout_input = nn.Dropout(dropout_prob)  # Dropout on the input
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        self.fc1 = nn.Linear(hidden_size3, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout before the dense layers
        # self.l1_lambda = l1_lambda

    def forward(self, x):
        batch_size = x.size(0)
        h0_1 = torch.zeros(1, batch_size, 64).to(x.device)  # Initialize with zeros
        c0_1 = torch.zeros(1, batch_size, 64).to(x.device)
        h0_2 = torch.zeros(1, batch_size, 128).to(x.device)
        c0_2 = torch.zeros(1, batch_size, 128).to(x.device)
        h0_3 = torch.zeros(1, batch_size, 64).to(x.device)
        c0_3 = torch.zeros(1, batch_size, 64).to(x.device)

        out    = self.norm(x)  # Apply normalization
        out, _ = self.lstm1(out, (h0_1, c0_1))
        out    = self.relu(out)
        out, _ = self.lstm2(out, (h0_2, c0_2))
        out    = self.relu(out)
        out, _ = self.lstm3(out, (h0_3, c0_3))
        out    = self.dropout(out)  # Apply dropout before the dense layers
        out    = self.relu(out[:, -1, :])  # Take the last output for the next dense layer

        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.softmax(self.fc3(out))

        return out


class LSTMAmanda(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=512, dropout_prob=0.4, l1_lambda=0.001):
        super(LSTMModel, self).__init__()
        self.norm = nn.LayerNorm(input_size)  # Apply normalization to the input
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout before the dense layers
        self.l1_lambda = l1_lambda

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(x.device)  # Initialize with zeros
        c0 = torch.zeros(1, batch_size, self.lstm.hidden_size).to(x.device)

        # Criação da máscara
        mask = (x != 0).float().unsqueeze(-1).to(x.device)

        out    = self.norm(x)  # Apply normalization
        out    = out * mask
        out, _ = self.lstm(out, (h0, c0))
        out    = self.relu(out)
        out    = self.dropout(out)
        out    = self.softmax(self.fc(out))

        return out
