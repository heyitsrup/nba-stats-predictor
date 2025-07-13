from torch import tensor, float32
from os import walk, path

def customTransform(sample):
    inputData = tensor([
        sample['PTS'], 
        sample['REB'], 
        sample['AST'], 
        sample['STL'], 
        sample['BLK']
    ], dtype=float32)
    
    target = inputData.clone()
    
    return inputData, target

def getJSONFilePaths(directory):
    JSONFilePaths = []
    for root, dir, files in walk(directory):
        for file in files:
            if file.endswith(".json"):
                relPath = path.join(root, file)
                JSONFilePaths.append(path.normpath(relPath))
    return JSONFilePaths