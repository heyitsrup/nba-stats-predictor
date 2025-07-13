from json import load
from numpy import array

def JSONToNumpy(json_file):
    with open(json_file, 'r') as f:
        data = load(f)
    
    metricsList = []
    
    for game in data:
        metrics = [
            game['PTS'],
            game['REB'],
            game['AST'],
            game['STL'],
            game['BLK']
        ]
        metricsList.append(metrics)
    
    metricsArray = array(metricsList)
    
    return metricsArray