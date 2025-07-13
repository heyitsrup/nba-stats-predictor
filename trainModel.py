from torch import no_grad, device, cuda, nn, optim, save
from defineModel import LSTMModel
from loadPlayerData import trainLoaders, valLoader
from tqdm import tqdm

def testModel(model, loader, criterion):
    model.eval()
    with no_grad():
        totalLoss = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            totalLoss += loss.item()
    return totalLoss / len(loader)

def trainModel(model, NUM_EPOCHS = 40, lr = 0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0)

    for epoch in range(NUM_EPOCHS):
        model.train()
        print(f'==> Epoch { epoch + 1 }')
        
        for loader in trainLoaders:
            for inputs, targets in tqdm(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs.unsqueeze(1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        lossVal = testModel(model=model, loader=valLoader, criterion=criterion)
        print(f'Validation Loss: {lossVal:.3f}')

    save(model.state_dict(), f"models/trained_lstm.pth")
    print(f"✅ Model saved as trained_lstm.pth")
    
device = device("cuda" if cuda.is_available() else f"cpu")
model = LSTMModel().to(device=device)
trainModel(model=model)