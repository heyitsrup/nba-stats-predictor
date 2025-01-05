import os, subprocess, torch
from flask import Flask, json, request, jsonify
from flask_cors import CORS
import numpy as np
import torch.nn as nn

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CORS(app)

@app.route("/api/process-player-data/", methods=['POST'])
def process_player_data():
    try:
        data = request.get_json()
        player_name = data.get('player_name')
        if not player_name:
            return jsonify({'error': 'Player name is required'}), 400

        result = subprocess.run(
            ['python', 'get_stats.py', '--player_name', player_name],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            output = result.stdout.strip().splitlines()

            # Run another subprocess command
            train_command = [
                'python', 
                os.path.join(r'C:\Users\singh\Documents\Programming\NBA Player Performance Prediction\performance_predictor_backend', 'train_model.py'), 
                player_name
            ]

            # Run the subprocess
            train_result = subprocess.run(train_command, capture_output=True, text=True)

            if train_result.returncode == 0:
                return jsonify({'success': True, 'output': train_result.stdout.strip().splitlines()})
            else:
                train_error_message = train_result.stderr.strip().splitlines()
                return jsonify({'error': 'Training model failed', 'details': train_error_message}), 500
        else:
            error_message = result.stderr.strip().splitlines()
            return jsonify({'error': 'Failed to process player data', 'details': error_message}), 500

    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/get-player-id/', methods=['GET'])
def get_player_id():
    try:
        # Extract 'playerName' from query parameters
        player_name = request.args.get('playerName')
        if not player_name:
            return jsonify({'error': 'Player name is required'}), 400

        # Find player ID using NBA API
        player = player.find_players_by_full_name(player_name)
        if not player:
            return jsonify({'error': 'Player not found'}), 404

        player_id = player[0]['id']
        response = {'playerId': player_id}
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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

@app.route('/api/predict/', methods=['GET'])
def predict_score():
    try:
        player_name = request.args.get('playerName', '').replace(' ', '_')
        if not player_name:
            return jsonify({'error': 'Player name is required'}), 400

        # Path to the directory containing player data
        directory = r'C:\Users\singh\Documents\Programming\NBA Player Performance Prediction\performance_predictor_backend\Player_Data'
        
        # Get JSON file paths for the player
        json_file_paths = get_json_file_paths(directory, player_name)
        if not json_file_paths:
            return jsonify({'error': f'No JSON files found for {player_name}'}), 404
        
        # Load the latest JSON file
        with open(json_file_paths[-1], 'r') as f:
            data = json.load(f)

        # Extract game metrics
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
        
        # Convert metrics to a NumPy array
        historical_data = np.array(metrics_list)

        # Load the trained model
        model = LSTMModel(5, 128, 5).to(device)
        state_dict = torch.load(r'C:\Users\singh\Documents\Programming\NBA Player Performance Prediction\performance_predictor_backend\Trained_Model.pth')
        model.load_state_dict(state_dict)
        model.eval()

        # Calculate average metrics
        avg_metrics = np.mean(historical_data, axis=0)

        # Prepare input for the model
        input_data = torch.tensor(avg_metrics, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Make predictions
        with torch.no_grad():
            outputs = model(input_data)
            predictions = outputs.cpu().numpy()

        # Ensure predictions are non-negative and rounded
        prediction = np.maximum(np.round(predictions.flatten()), 0)

        # Return predictions as a response
        response = {'prediction': prediction.tolist()}
        return jsonify(response)

    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)