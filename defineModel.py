from torch import nn, zeros

class LSTMModel(nn.Module):
    def __init__(self, inputSize=5, hiddenSize=128, outputSize=5, numLayers=1):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.numLayers= 1
        self.lstm = nn.LSTM(inputSize, hiddenSize, numLayers, batch_first=True)
        self.fc = nn.Linear(hiddenSize, outputSize)
    
    def forward(self, x):
        h0 = zeros(self.numLayers, x.size(0), self.hiddenSize).to(x.device)
        c0 = zeros(self.numLayers, x.size(0), self.hiddenSize).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        return self.fc(out[:, -1, :])