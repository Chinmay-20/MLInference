import torch
from torch.utils.data import Dataset, DataLoader
import os

# Dataset: base class to define dataset. 
# PyTorch get information about: how many data are there and how to get them

# Dataloader: handle batching, shuffle.

class RandomDataset(Dataset):
    def __init__(self, data_tensor, labels_tensor):
        self.data = data_tensor
        self.labels = labels_tensor

    def __len__(self):
        return len(self.data)

    # For example: dataset[0] -> (input, label)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# cgf = HYDRA configuration
# Useful to take: 
'''
cfg.paths.processed_data
cfg.model.input_size
cfg.training.batch_size
'''
def get_dataloaders(cfg):
    """
    Load processed dataset (saved as .pt) and get DataLoader.
    """
    data_path = cfg.paths.processed_data
    
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        dataset_dict = torch.load(data_path)
        X = dataset_dict['data']
        y = dataset_dict['labels']
    else:
        print(f"Data not found at {data_path}. Generating synthetic data...")
        # Generate data suitable for SimpleMLP (input 10)
        # 1000 samples
        X = torch.randn(1000, cfg.model.input_size)
        # Simple target: input sum + noise. Simple regression problem.
        # For example: [1,2,3] -> 6 + noise
        y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(1000, 1)
        
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        torch.save({'data': X, 'labels': y}, data_path)
        print(f"Data saved to {data_path}")

    # Create Dataset and DataLoader
    dataset = RandomDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
    
    return dataloader