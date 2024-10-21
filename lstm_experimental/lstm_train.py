import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Tuple

from models import AdaptiveComputationTimeWrapper
from LSTMwrapper import AdaptiveLSTMWrapper

# Define your LSTM model
class MyLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size):
        # Initialize hidden and cell states to zeros
        h_0 = torch.zeros(1, batch_size, self.hidden_size)
        c_0 = torch.zeros(1, batch_size, self.hidden_size)
        return (h_0, c_0)

    def forward(self, input, hidden):
        # input shape: (batch, input_size)
        input = input.unsqueeze(0)  # (1, batch, input_size)
        output, hidden = self.lstm(input, hidden)
        output = output.squeeze(0)  # (batch, hidden_size)
        output = self.fc(output)  # (batch, output_size)
        return output, hidden

# Instantiate your model
input_size = 10
hidden_size = 20
output_size = 5
lstm_model = MyLSTMModel(input_size, hidden_size, output_size)

# Wrap the model with ACT
act_lstm = AdaptiveLSTMWrapper(
    model=lstm_model,
    compute_hidden=lambda hidden: hidden[0].squeeze(0),  # Extract h_t
    time_penalty=0.1,
    initial_halting_bias=-1.0,
    ponder_epsilon=1e-2,
    time_limit=10,
)

# Example input
batch_size = 3
inputs = torch.randn(batch_size, input_size)
hidden = lstm_model.init_hidden(batch_size)

# Forward pass
final_output, final_hidden, ponder_cost, ponder_steps = act_lstm(inputs, hidden)

print("Final Output:", final_output)
print("Ponder Cost:", ponder_cost)
print("Ponder Steps:", ponder_steps)
