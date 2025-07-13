from torch import no_grad, nn, optim, save
from tqdm import tqdm

def testModel(model, criterion, loader, device):
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

def trainModel(model, trainLoaders, valLoader, device, playerName, NUM_EPOCHS = 40, lr = 0.001):
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
        
        lossVal = testModel(model=model, criterion=criterion, loader=valLoader, device=device)
        print(f'Validation Loss: {lossVal:.3f}')

    save(model.state_dict(), f"models/{ playerName.replace(' ', '_') }_trained_lstm.pth")
    print(f"✅ Model saved as { playerName.replace(' ', '_') }_trained_lstm.pth")