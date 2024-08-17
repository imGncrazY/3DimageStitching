import torch
import os
import numpy as np
from torch.utils.data import Dataset
from typing import Any

class imgStitching(Dataset):
    def __init__(
        self,
        data_dir: str,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.npy_names = sorted([name for name in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, name)) and name.endswith('.npy')])
    
    def __getitem__(self, index) -> Any:
        np_file = np.load(os.path.join(self.data_dir, self.npy_names[index]))
        np_data = np_file[0,:]
        np_target = np_file[1,:]
        return np_data, np_target
    
    def __len__(self) -> int:
        return len(self.npy_names)

