# myapp/management/commands/train_model.py
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from django.core.management.base import BaseCommand

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_json_file_paths(directory):
    json_file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_file_paths.append(os.path.join(root, file))
    return json_file_paths

class Command(BaseCommand):
    help = 'Train the PyTorch model on game data JSON files'

    def add_arguments(self, parser):
        parser.add_argument('player_name', type=str, help='Name of the player whose data to train on')

    def handle(self, *args, **options):
        player_name = options['player_name']

        # Set parameters
        BATCH_SIZE = 2
        NUM_EPOCHS = 40
        
        model = LSTMModel(5,128,5).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.95), weight_decay=0)
        criterion = nn.MSELoss()

        # Load data
        player_data_directory = os.path.join(r'C:\Users\singh\Documents\Programming\NBA Player Performance Prediction\performance_predictor_backend\Player_Data', player_name.replace(' ', '_'))
        json_file_paths = get_json_file_paths(player_data_directory)

        train_datasets = [JSONDataset(json_file, transform=custom_transform) for json_file in json_file_paths[:-2]]
        train_loaders = [DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) for dataset in train_datasets]

        val_dataset = JSONDataset(json_file_paths[-2], transform=custom_transform)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Training loop
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
            
            # Validation
            val_loss = self.test(val_loader, model, criterion)
            print(f'Validation Loss: {val_loss:.3f}')

        torch.save(model.state_dict(), r'C:\Users\singh\Documents\Programming\NBA Player Performance Prediction\performance_predictor_backend\Trained_Model.pth')

    def test(self, loader, model, criterion):
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
