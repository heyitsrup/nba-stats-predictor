from streamlit import title, text_input, button, warning, success, write, columns, empty, progress
from fetchPlayerData import fetchPlayerData
from processPlayerData import getJSONFilePaths
from loadPlayerData import loadPlayerData
from torch import device, cuda, load
from defineModel import LSTMModel
from trainModel import trainModel
from JSONToNumpy import JSONToNumpy
from predictNextGame import predictNextGame
from os.path import exists, join, isdir
from testMetrics import testMetrics
from plotPredictions import plotPredictions

title("🏀 NBA Athlete Performance Prediction ML Model")

playerName = text_input("Enter NBA Player Name")

progress_bar = progress(0)
log_area = empty()
log_text = ""

def log_fn(msg):
    global log_text
    log_text += msg + "\n"
    log_area.text_area("Training Log", log_text, height=100)

def progress_fn(p):
    progress_bar.progress(p)

def handlePlayerFetch(playerName):
    data_dir = join("data", "raw", playerName)
    model_path = join("models", f"{ playerName }.pth")

    device_used = device("cuda" if cuda.is_available() else "cpu")
    model = LSTMModel().to(device=device_used)

    JSONFilePaths = getJSONFilePaths(playerName=playerName)

    if exists(model_path):
        # ✅ Load the model
        model.load_state_dict(load(model_path, map_location=device_used))
        success(f"Loaded existing model for {playerName}")

        # ✅ Load data only for test metrics & prediction
        trainLoaders, valLoader, testLoader = loadPlayerData(JSONFilePaths=JSONFilePaths)

    else:
        # 🚨 Only fetch data if none exists
        if not isdir(data_dir) or len(JSONFilePaths) == 0:
            warning("No historical data found for training.")
            fetchPlayerData(playerName=playerName)
            success(f"Fetched new data for {playerName}")
            JSONFilePaths = getJSONFilePaths(playerName=playerName)

        # ✅ Now load data and train
        trainLoaders, valLoader, testLoader = loadPlayerData(JSONFilePaths=JSONFilePaths)

        trainModel(
            model=model,
            trainLoaders=trainLoaders,
            valLoader=valLoader,
            device=device_used,
            playerName=playerName,
            log_fn=log_fn,
            progress_fn=progress_fn
        )
        success(f"Model trained and saved for {playerName}")

    # ✅ Evaluate and display predictions
    mse, mae, r2, trueLabelsNp, predictionsNp = testMetrics(model=model, device=device_used, loader=testLoader)

    historicalData = JSONToNumpy(JSONFilePaths[-1])
    predictedMetrics = predictNextGame(model, historicalData, device_used)

    labels = ["Points", "Rebounds", "Assists", "Steals", "Blocks"]
    cols = columns(5)
    for col, value, label in zip(cols, predictedMetrics, labels):
        col.metric(label=label, value=f"{value:.1f}")
    
    write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    write(f"**R-squared (R²):** {r2 * 100:.2f}%")

    plotPredictions(trueLabelsNp=trueLabelsNp, predictionsNp=predictionsNp)

if button("Fetch Player Data"):
    if playerName.strip() == "":
        warning("Please enter a player name.")
    else:
        handlePlayerFetch(playerName.strip())