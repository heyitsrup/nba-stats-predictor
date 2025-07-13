from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch import no_grad
from loadPlayerData import testLoader
from numpy import array

def testMetrics(model, device):
    model.eval()
    predictions = []
    trueLabels = []

    with no_grad():
        for inputs, targets in testLoader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(1) 
            outputs = model(inputs)
            
            targetsNp = targets.cpu().numpy()
            outputsNp = outputs.cpu().numpy()
            
            trueLabels.extend(targetsNp)
            predictions.extend(outputsNp)

    predictionsNp = array(predictions)
    trueLabelsNp = array(trueLabels)

    mse = mean_squared_error(trueLabelsNp, predictionsNp)
    mae = mean_absolute_error(trueLabelsNp, predictionsNp)
    r2 = r2_score(trueLabelsNp, predictionsNp)

    print("Mean Squared Error (MSE):", mse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R2):", r2)

    return trueLabelsNp, predictionsNp