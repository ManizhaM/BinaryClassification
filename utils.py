from collections.abc import Iterable
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader 
import torch.nn as nn


def get_text_corpus(texts: Iterable[list]) -> list:
    corpus = [word for text in texts for word in text]
    return list(set(corpus))

# Embedding dict
def build_emb_dict(corpus: list, model) -> dict:
    emb_dict = {}
    for word in corpus:
        emb_dict[word] = model.wv[word]
    return emb_dict

def convert_to_emb(X: np.ndarray, emb_dict: dict) -> np.ndarray:
    return [np.array([emb_dict[word] for word in sample]) for sample in X]

class FastTextDataset(Dataset):
    def __init__(self, X: list, y: np.ndarray, max_seq_length: float = 400):
        self.X = X
        self.y = y
        self.max_seq_length = max_seq_length
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        X = torch.tensor(self.X[index]).float()
        y = torch.tensor(self.y[index]).float()
        seq_length = torch.tensor(self.X[index].shape[0])
        
        X = nn.functional.pad(X, (0, 0, 0, self.max_seq_length - X.shape[0]))
        return X, y, seq_length
    