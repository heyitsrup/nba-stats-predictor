from numpy import mean, maximum, round
from torch import no_grad, tensor, float32, load, device, cuda
from defineModel import LSTMModel
from JSONToNumpy import JSONToNumpy
from processPlayerData import getJSONFilePaths

device = device("cuda" if cuda.is_available() else f"cpu")
model = LSTMModel().to(device)
model.load_state_dict(load("models/trained_lstm.pth", map_location=device))

def predictNextGame(model, historicalData):
    model.eval()
    
    avg_metrics = mean(historicalData, axis=0)
    
    input_data = tensor(avg_metrics, dtype=float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with no_grad():
        outputs = model(input_data)
        predictions = outputs.cpu().numpy()
    
    return maximum(round(predictions.flatten()), 0)

JSONFilePaths = getJSONFilePaths("./data/raw")
historicalData = JSONToNumpy(JSONFilePaths[-1])
predictedMetrics = predictNextGame(model, historicalData)
print(f"Predicted Metrics for Next Game based on Historical Data: { predictedMetrics } ")