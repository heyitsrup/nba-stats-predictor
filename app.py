from streamlit import title, text_input, button, warning, success, write
from fetchPlayerData import fetchPlayerData
from processPlayerData import getJSONFilePaths
from loadPlayerData import loadPlayerData
from torch import device, cuda, load
from defineModel import LSTMModel
from trainModel import trainModel
from JSONToNumpy import JSONToNumpy
from predictNextGame import predictNextGame

title("🏀 NBA Player Performance Predictor")

playerName = text_input("Enter NBA Player Name")

if button("Fetch Player Data"):
    if playerName.strip() == "":
        warning("Please enter a player name.")
    else:
        fetchPlayerData(playerName=playerName)
        success(f"Data fetched for {playerName}")

        JSONFilePaths = getJSONFilePaths()
        trainLoaders, valLoader, testLoader = loadPlayerData(JSONFilePaths=JSONFilePaths)

        device = device("cuda" if cuda.is_available() else f"cpu")
        model = LSTMModel().to(device=device)

        trainModel(model=model, trainLoaders=trainLoaders, valLoader=valLoader, device=device, playerName=playerName)
        
        trainedModel = model.load_state_dict(load("models/trained_lstm.pth", map_location=device))
        historicalData = JSONToNumpy(JSONFilePaths[-1])
        predictedMetrics = predictNextGame(model, historicalData, device)
        write(f"Predicted Metrics for Next Game based on Historical Data: { predictedMetrics } ")
