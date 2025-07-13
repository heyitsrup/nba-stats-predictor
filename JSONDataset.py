from torch.utils.data import Dataset
from json import load

class JSONDataset(Dataset):
    def __init__(self, JSONFile, transform=None):
        with open(JSONFile, 'r', encoding='utf-8') as f:
            self.data = load(f)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if not self.data:
            raise IndexError("Dataset is empty or not loaded properly.")
        
        sample = self.data[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample