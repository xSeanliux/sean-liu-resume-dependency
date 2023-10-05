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


def get_context(buf_str, stk_str): 
    return f'context: {buf_str} ; {stk_str}</s>options: discard: 0, merge: 1, pop: 2, subordinate: 3'

def get_prompt_item(sample): 
    source_ids = sample["input_ids"].squeeze()
    target_ids = sample["input_ids"].squeeze()

    src_mask    = sample["attention_mask"].squeeze()  # might need to squeeze
    target_mask = sample["attention_mask"].squeeze()  # might need to squeeze

    return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
  

class ResumePromptDataset(Dataset):
    def __init__(self, tokenizer, data_dir, max_len=512):
        self.tokenizer = tokenizer
        self.df = pd.read_json(data_dir)

        self.max_len = max_len
        
        self.inputs = []
        self.targets = []
        # self.elements = []

        self._build()
    def __len__(self):
        return len(self.df)
    
    def _build(self):
        for _, element in self.df.iterrows():
            
            # pos = torch.floor(torch.Tensor([element.lbuf, element.rbuf, element.lstk, element.rstk])).long()
            # sty = torch.Tensor([element.italbuf, element.boldbuf, element.italstk, element.boldstk]).long()
            context = get_context(element.buf_str, element.stk_str)
            inp_ids = self.tokenizer(
                [context], return_tensors="pt",
                max_length=self.max_len, pad_to_max_length=True
            )
            out_ids = self.tokenizer(
                [str(element.type)], return_tensors="pt",
                max_length=2, pad_to_max_length=True
            )
            self.inputs.append(inp_ids)
            self.targets.append(out_ids)

    def __getitem__(self, index):
        return get_item(self.inputs[index])

