import numpy as np
import torch

EMB_DIM = 50
NUM_HIDDEN_NODES = 64
NUM_OUTPUT_NODES = 1
NUM_LAYERS = 2
BIDIRECTION = True
DROPOUT = 0.2
BATCH_SIZE = 32

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
