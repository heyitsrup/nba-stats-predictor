from torch import load, device, cuda
from defineModel import LSTMModel
from testMetrics import testMetrics
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, legend, show
from numpy import maximum, round

device = device("cuda" if cuda.is_available() else f"cpu")
model = LSTMModel().to(device)
model.load_state_dict(load("models/trained_lstm.pth", map_location=device))
trueLabelsNp, predictionsNp = testMetrics(model, device)

for i, metric in enumerate(['PTS', 'REB', 'AST', 'STL', 'BLK']):
    figure(figsize=(10, 6))
    plot(trueLabelsNp[:, i], label='Actual', marker='o')
    plot(maximum(round(predictionsNp[:, i]), 0), label='Predicted', marker='x')
    title(f'{metric}: Predicted vs Actual')
    xlabel('Game')
    ylabel(metric)
    legend()
    show()