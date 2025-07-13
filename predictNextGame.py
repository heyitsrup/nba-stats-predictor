from numpy import mean, maximum, round
from torch import no_grad, tensor, float32

def predictNextGame(model, historicalData, device):
    model.eval()
    
    avg_metrics = mean(historicalData, axis=0)
    
    input_data = tensor(avg_metrics, dtype=float32).unsqueeze(0).unsqueeze(0).to(device)
    
    with no_grad():
        outputs = model(input_data)
        predictions = outputs.cpu().numpy()
    
    return maximum(round(predictions.flatten()), 0)