from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch import no_grad
from numpy import array

def testMetrics(model, device, loader):
    model.eval()
    predictions = []
    trueLabels = []

    with no_grad():
        for inputs, targets in loader:
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

    return mse, mae, r2, trueLabelsNp, predictionsNp