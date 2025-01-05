import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset for JSON data
class JSONDataset(Dataset):
    def __init__(self, json_file, transform=None):
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.data:
            raise IndexError("Dataset is empty or not loaded properly.")
        
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# LSTM Model
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

# Data transformation function
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

# Helper function to get JSON file paths
def get_json_file_paths(directory):
    return [
        os.path.join(root, file) 
        for root, dirs, files in os.walk(directory) 
        for file in files 
        if file.endswith(".json")
    ]

# Training and validation functions
def train_model(player_name, num_epochs=40, batch_size=2):
    # Set parameters
    player_data_directory = os.path.join(
        r'C:\Users\singh\Documents\Programming\NBA Player Performance Prediction\performance_predictor_backend\Player_Data', 
        player_name.replace(' ', '_')
    )
    
    # Load data
    json_file_paths = get_json_file_paths(player_data_directory)
    if len(json_file_paths) < 2:
        raise FileNotFoundError(f"Not enough JSON files found for player: {player_name}")

    train_datasets = [JSONDataset(json_file, transform=custom_transform) for json_file in json_file_paths[:-2]]
    train_loaders = [DataLoader(dataset, batch_size=batch_size, shuffle=True) for dataset in train_datasets]

    val_dataset = JSONDataset(json_file_paths[-2], transform=custom_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = LSTMModel(5, 128, 5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.95), weight_decay=0)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        print(f'==> Epoch {epoch + 1}')
        for loader in train_loaders:
            for inputs, targets in tqdm(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(1)  # Add sequence dimension
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        # Validation
        val_loss = test_model(val_loader, model, criterion)
        print(f'Validation Loss: {val_loss:.3f}')
    
    # Save trained model
    torch.save(model.state_dict(), r'C:\Users\singh\Documents\Programming\NBA Player Performance Prediction\performance_predictor_backend\Trained_Model.pth')

def test_model(loader, model, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(1)  # Add sequence dimension
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(loader)

# Flask CLI Command
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the LSTM model on player data")
    parser.add_argument('player_name', type=str, help="Name of the player whose data to train on")
    args = parser.parse_args()

    try:
        train_model(args.player_name)
        print(f"Model trained successfully for player: {args.player_name}")
    except Exception as e:
        print(f"Error: {e}")
