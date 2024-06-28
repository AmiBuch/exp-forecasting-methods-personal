import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import math
from sklearn.metrics import r2_score

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model))
        out = torch.matmul(attn_weights, V)
        return out

class TimeSeriesForecast(nn.Module):
    def __init__(self, input_len, seq_len, d_model, forecast_len):
        super(TimeSeriesForecast, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_chunks = input_len // seq_len
        self.input_linear = nn.Linear(1, d_model)  # Map 1D input to d_model dimensions
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        self.self_attn = SelfAttention(d_model)
        self.fc = nn.Linear(self.num_chunks * seq_len * d_model, forecast_len)

    def forward(self, x):
        batch_size, _, _ = x.size()
        x = self.input_linear(x)
        x = x.view(batch_size, self.num_chunks, self.seq_len, self.d_model)
        x = x.view(batch_size * self.num_chunks, self.seq_len, self.d_model)
        x = self.pos_encoder(x)
        x = self.self_attn(x)
        x = x.view(batch_size, self.num_chunks, self.seq_len * self.d_model)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x

# Load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
air_passengers = pd.read_csv(url, index_col='Month', parse_dates=True)

# Convert the 'Passengers' column to a numpy array
raw_seq = air_passengers['Passengers'].values

# Parameters
input_len = 48
forecast_len = 12
d_model = 100
seq_len = 8

# Split the dataset
train_seq = raw_seq[:132]
test_seq = raw_seq[132 - input_len:144]  # Include 48 values before the last 12 for input

# Create input-output pairs for training
def create_sequences(data, input_len, forecast_len):
    inputs = []
    targets = []
    for i in range(len(data) - input_len - forecast_len + 1):
        input_seq = data[i:i+input_len]
        target_seq = data[i+input_len:i+input_len+forecast_len]
        inputs.append(input_seq)
        targets.append(target_seq)
    return np.array(inputs), np.array(targets)

train_inputs, train_targets = create_sequences(train_seq, input_len, forecast_len)

# Reshape inputs to (batch_size, input_len, 1) since each timestep is a single value
train_inputs = train_inputs.reshape((train_inputs.shape[0], input_len, 1))

# Convert to PyTorch tensors
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_targets = torch.tensor(train_targets, dtype=torch.float32)

# Model instantiation
model = TimeSeriesForecast(input_len=input_len, seq_len=seq_len, d_model=d_model, forecast_len=forecast_len)

# Check if GPU is available and move model to GPU if it is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Move data to the same device as model
train_inputs = train_inputs.to(device)
train_targets = train_targets.to(device)

# Loss and optimizer
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 3000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(train_inputs)
    loss = criterion(output, train_targets)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prepare the test input
test_input = test_seq[:input_len].reshape(1, input_len, 1)
test_input = torch.tensor(test_input, dtype=torch.float32).to(device)

# Predict the last 12 values
model.eval()
with torch.no_grad():
    test_output = model(test_input)
    print("Predicted values:", test_output.cpu().numpy().flatten())
    print("Actual values:", test_seq[input_len:])
    # Calculate MAE
    mae = np.mean(np.abs(test_output.cpu().numpy().flatten() - test_seq[input_len:]))
    print(f"MAE: {mae}")

    # Calculate R^2
    r2 = r2_score(test_seq[input_len:], test_output.cpu().numpy().flatten())
    print(f"R^2: {r2}")
