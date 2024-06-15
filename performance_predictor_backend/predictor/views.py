# views.py
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import subprocess
import json

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@csrf_exempt
@require_POST
def process_player_data(request):
    try:
        data = json.loads(request.body)
        player_name = data.get('player_name')
        if not player_name:
            return JsonResponse({'error': 'Player name is required'}, status=400)

        # Run Python script using subprocess
        result = subprocess.run(
            ['python', 'get_stats.py', '--player_name', player_name],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            output = result.stdout.strip().splitlines()  # Example: process output if needed
            subprocess.run(
                ['python', 'manage.py', 'train_model', player_name],
            capture_output=True,
                text=True
            )
            return JsonResponse({'success': True, 'output': output})
        else:
            error_message = result.stderr.strip().splitlines()  # Example: handle error messages
            return JsonResponse({'error': 'Failed to process player data', 'details': error_message}, status=500)

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    
def get_json_file_paths(directory, player_name):
    player_directory = os.path.join(directory, player_name)
    json_file_paths = []
    for root, dirs, files in os.walk(player_directory):
        for file in files:
            if file.endswith(".json"):
                json_file_paths.append(os.path.join(player_directory, file))
    return json_file_paths

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

def predict_score(request):
    try:
        playerName = (request.GET.get('playerName')).replace(' ', '_')
        directory = r'C:\Users\singh\Documents\Programming\NBA Player Performance Prediction\performance_predictor_backend\Player_Data'
        
        json_file_paths = get_json_file_paths(directory, playerName)
        
        if not json_file_paths:
            return JsonResponse({'error': f'No JSON files found for {playerName}'}, status=404)
        
        with open(json_file_paths[-1], 'r') as f:
            data = json.load(f)
    
        metrics_list = []
    
        for game in data:
            metrics = [
                game['PTS'],
                game['REB'],
                game['AST'],
                game['STL'],
                game['BLK']
            ]

            metrics_list.append(metrics)

        historical_data = np.array(metrics_list)

        model = LSTMModel(5,128,5).to(device)
        state_dict = torch.load(r'C:\Users\singh\Documents\Programming\NBA Player Performance Prediction\performance_predictor_backend\Trained_Model.pth')
        model.load_state_dict(state_dict)
        model.eval()
        avg_metrics = np.mean(historical_data, axis=0)  # Calculate average along rows (axis=0)
    
        input_data = torch.tensor(avg_metrics, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
        with torch.no_grad():
            outputs = model(input_data)
            predictions = outputs.cpu().numpy()
        
        prediction = np.maximum(np.round(predictions.flatten()), 0)
        response = {'prediction': prediction.tolist()}
        return JsonResponse(response)
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)