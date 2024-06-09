import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import r2_score
import torch.optim as optim
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
air_passengers = pd.read_csv(url, index_col='Month', parse_dates=True)
raw_seq = air_passengers['Passengers'].values
# Data preparation
n_steps = 48
X, y = split_sequence(raw_seq[:-12], n_steps)
# Generate positional encodings
def get_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pe, dtype=torch.float32)

# Example usage:
seq_len = 48  # Length of the input sequence
d_model = 1  # Dimension of each data point (assuming a single feature per timestamp)
positional_encoding = get_positional_encoding(seq_len, d_model)
# Model definition
class MLPWithPositionalEncoding(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        super(MLPWithPositionalEncoding, self).__init__()
        self.seq_len = seq_len
        self.d_model = input_size // seq_len
        
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

        # Positional encoding
        self.positional_encoding = get_positional_encoding(seq_len, self.d_model)

    def forward(self, x):
        # Reshape x to (batch_size, seq_len, d_model) if necessary
        batch_size = x.shape[0]
        x = x.view(batch_size, self.seq_len, self.d_model)
        
        # Apply positional encoding
        x = x * self.positional_encoding

        # Flatten the input again
        x = x.view(batch_size, -1)

        # Pass through MLP layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model
input_size = 48  # Number of input data points
hidden_size = 100  # Size of hidden layer
output_size = 1  # Single output value
seq_len = 48  # Length of input sequence

model = MLPWithPositionalEncoding(input_size, hidden_size, output_size, seq_len)

# Define a loss function and an optimizer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


inputs = torch.from_numpy(X).float()
targets = torch.from_numpy(y).float()
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=12)

# Training loop
for epoch in range(3000):
    for inputs, targets in dataloader:
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Prediction
input_hours = 48
output_hours = 12
start_idx = 83
final_pred = []

for i in range(output_hours):
    x_input = raw_seq[start_idx+i:start_idx+input_hours+i]
   
    x_input = torch.from_numpy(np.array(x_input)).float()
    yhat = model(x_input.unsqueeze(0)).item()
    final_pred.append(yhat)

print("Predicted values:")
print(final_pred)
print("Actual values:")
actual_values = raw_seq[start_idx+input_hours:start_idx+input_hours+output_hours]
print(actual_values)

# Calculate MAE
mae = np.mean(np.abs(np.array(final_pred) - actual_values))
print(f"MAE: {mae}")

# Calculate R^2
r2 = r2_score(actual_values, final_pred)
print(f"R^2: {r2}")
