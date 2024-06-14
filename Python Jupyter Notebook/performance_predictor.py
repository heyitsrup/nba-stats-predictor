from tqdm import tqdm
import numpy as np
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class JSONDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

def custom_transform(sample):
    input_data = torch.tensor([
        sample['PTS'], 
        sample['REB'], 
        sample['AST'], 
        sample['STL'], 
        sample['BLK']
    ], dtype=torch.float32)
    
    target = input_data.clone()
    
    return input_data, target

NUM_EPOCHS = 40
BATCH_SIZE = 2
lr = 0.001

def get_json_file_paths(directory):
    json_file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_file_paths.append(os.path.relpath(os.path.join(root, file), directory))
    return json_file_paths

json_file_paths = get_json_file_paths(".")
print("Relative JSON file paths:")
print(json_file_paths)

train_datasets = [JSONDataset(json_file, transform=custom_transform) for json_file in json_file_paths[0:-2]]
train_loaders = [DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) for dataset in train_datasets]

val_dataset = JSONDataset(json_file_paths[-2], transform=custom_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = JSONDataset(json_file_paths[-1], transform=custom_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel(5,128,5).to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0)

def test(loader, model, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

for epoch in range(NUM_EPOCHS):
    model.train()
    print(f'==> Epoch {epoch+1}')
    
    for loader in train_loaders:
        for inputs, targets in tqdm(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    val_loss = test(val_loader, model, criterion)
    print(f'Validation Loss: {val_loss:.3f}')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.unsqueeze(1) 
        outputs = model(inputs)
        
        targets_np = targets.cpu().numpy()
        outputs_np = outputs.cpu().numpy()
        
        true_labels.extend(targets_np)
        predictions.extend(outputs_np)

predictions = np.array(predictions)
true_labels = np.array(true_labels)

mse = mean_squared_error(true_labels, predictions)
mae = mean_absolute_error(true_labels, predictions)
r2 = r2_score(true_labels, predictions)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)

import matplotlib.pyplot as plt

for i, metric in enumerate(['PTS', 'REB', 'AST', 'STL', 'BLK']):
    plt.figure(figsize=(10, 6))
    plt.plot(true_labels[:, i], label='Actual', marker='o')
    plt.plot(np.maximum(np.round(predictions[:, i]), 0), label='Predicted', marker='x')
    plt.title(f'{metric}: Predicted vs Actual')
    plt.xlabel('Game')
    plt.ylabel(metric)
    plt.legend()
    plt.show()

def find_game_index_by_date(target_game_date):
    with open(json_file_paths[5], 'r') as f:
        data = json.load(f)
    
    for index, game in enumerate(data):
        if game.get('GAME_DATE') == target_game_date:
            return index
    
    return None

def plot_predictions_vs_actuals_for_game(model, loader, device, game_date):
    model.eval()
    game_index = find_game_index_by_date(game_date)
    
    inputs, targets = next(iter(loader))
    inputs, targets = inputs[game_index].unsqueeze(0).unsqueeze(0).to(device), targets[game_index].unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
        predictions = outputs.cpu().numpy()
        true_labels = targets.cpu().numpy()
    
    rounded_predictions = np.maximum(np.round(predictions.flatten()), 0)

    plt.figure(figsize=(8, 5))
    plt.plot(true_labels.flatten(), label='Actual', marker='o')
    plt.plot(rounded_predictions, label='Predicted', marker='x')

    print("Predicted Values:", rounded_predictions)
    print("Actual Values:", true_labels.flatten())

    metric_names = ['PTS', 'REB', 'AST', 'STL', 'BLK'] 
    plt.xticks(np.arange(len(metric_names)), metric_names) 

    plt.title(f'Predicted vs Actual for Game {game_date}')
    plt.xlabel('Metric')
    plt.legend()
    plt.show()

plot_predictions_vs_actuals_for_game(model, test_loader, device, "2024-06-09")



