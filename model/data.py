from torch.utils.data import Dataset
from utils import label_idx
import pandas as pd 
import torch 


'''
The ResumeDataset class wraps the .json file generated from pkl_to_json.ipynb into a PyTorch Dataset.
'''
class ResumeDataset(Dataset):
    def __init__(self, data_dir):
        self.df = pd.read_json(data_dir)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        element = self.df.iloc[idx]
        pos = torch.floor(torch.Tensor([element.lbuf, element.rbuf, element.lstk, element.rstk])).long()
        sty = torch.Tensor([element.italbuf, element.boldbuf, element.italstk, element.boldstk]).long()
        return element.buf_str, element.stk_str, pos, sty, label_idx[element.type]

