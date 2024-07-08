import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, device):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # 隠れ層から隠れ層への線形変換
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )

        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1).to(self.device)
        hidden = self.i2h(combined)
        hidden = F.relu(hidden)

        for layer in self.hidden_layers:
            hidden = layer(hidden)
            hidden = F.relu(hidden)

        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(self.device)
