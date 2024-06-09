# univariate data preparation
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from numpy import array
import pandas as pd
from sklearn.metrics import r2_score
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
 X, y = list(), list()
 for i in range(len(sequence)):
 # find the end of this pattern
  end_ix = i + n_steps
  # check if we are beyond the sequence
  if end_ix > len(sequence)-1:
    break
 # gather input and output parts of the pattern
  seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
  X.append(seq_x)
  y.append(seq_y)
 return array(X), array(y)


# Load the dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
air_passengers = pd.read_csv(url, index_col='Month', parse_dates=True)


# Convert the 'Passengers' column to a numpy array
raw_seq = air_passengers['Passengers'].values


# choose a number of time steps
n_steps = 48
# split into samples
X, y = split_sequence(raw_seq[:-12], n_steps)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size=48, hidden_size=200, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)  
        self.fc5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Instantiate the model
model = MLP()
sequence_length = 48
inputs = X
targets = y


# Convert to PyTorch tensors and create a DataLoader
inputs = torch.from_numpy(np.array(inputs)).float()
targets = torch.from_numpy(np.array(targets)).float()
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=12)

# Define a loss function and an optimizer
criterion = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(3000):  # 100 epochs
    for inputs, targets in dataloader:
        # Forward pass
        

        
        outputs = model(inputs)
        #print(outputs)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, targets)
        #print(outputs.shape, targets.shape)
           
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
inputhours = 48
outputhours = 1
outputhour = 12
start_idx = 83
final_pred = []
yhat = raw_seq[131]
for part_start in range(83, 95):
   x_input = np.array(raw_seq[part_start:part_start+48])
   x_input[47] = yhat
   x_input = torch.from_numpy(np.array(x_input)).float()
   yhat = model(x_input).item()
   final_pred.append(yhat)
print("predicted values : ")
print(final_pred)
print("actual values : ")
actual_values = raw_seq[start_idx+inputhours:start_idx+(inputhours+outputhour)]
print(actual_values)

# Calculate MAE
mae = np.mean(np.abs(final_pred - actual_values))
print(f"MAE: {mae}")


# Calculate R^2
r2 = r2_score(actual_values, final_pred)
print(f"R^2: {r2}")
   