import glob

import pandas as pd
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

class CostumDataset(Dataset):
    def __init__(self, ml_method: str) -> None:
       self.data_location = 'Data/datasets/' + ml_method + '/'
       self.file_names = glob.glob(self.data_location + '*.pkl')
       self.ml_method = ml_method
           
    def __len__(self) -> int:
        return len(self.file_names)
    
    def __getitem__(self, index: int) -> Tuple: 
        data = pd.read_pickle(self.file_names[index])
        
        if self.ml_method in ['GNN', 'GNN_plus']:
            return data['N_electrons'].tolist(), data[self.ml_method].tolist()
        
        else:
            data_list = torch.tensor(data[self.ml_method])
            X, Y = torch.stack(data_list, dim=0)

            # X = torch.tensor(data_list[0], 
            #                  requires_grad = True)
            # Y = torch.tensor(data_list[1], 
            #                  requires_grad = True)
            return X, Y
    
