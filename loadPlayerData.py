from JSONDataset import JSONDataset
from torch.utils.data import DataLoader
from processPlayerData import getJSONFilePaths, customTransform

JSONFilePaths = getJSONFilePaths("./data/raw")

trainDatasets = [JSONDataset(json_file, transform=customTransform) for json_file in JSONFilePaths[0:-2]]
trainLoaders = [DataLoader(dataset, batch_size=2, shuffle=True) for dataset in trainDatasets]

valDataset = JSONDataset(JSONFilePaths[-2], transform=customTransform)
valLoader = DataLoader(valDataset, batch_size=2, shuffle=False)

testDataset = JSONDataset(JSONFilePaths[-1], transform=customTransform)
testLoader = DataLoader(testDataset, batch_size=2, shuffle=False)