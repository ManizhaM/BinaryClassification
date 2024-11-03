import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from gensim.models import FastText
from src.model.LSTMNet import LSTMNet
from src.train import train
from src.visualization.plot_metrics import plot_loss, plot_accuracy
from src.data.make_dataset import data_preprocessing, data_split
from cfg.params import *
from collections.abc import Iterable
from utils import get_text_corpus, build_emb_dict, convert_to_emb, FastTextDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    df = data_preprocessing()
    X_train, X_test, y_train, y_test = data_split(df)

    fasttext_model = FastText(vector_size=EMB_DIM, window=3, min_count=1)
    fasttext_model.build_vocab(corpus_iterable=X_train)
    fasttext_model.train(corpus_iterable=X_train, total_examples=len(X_train), epochs=10)

    embedding_dict = build_emb_dict(get_text_corpus(X_train), fasttext_model)

    X_train_emb = convert_to_emb(X_train, embedding_dict)
    X_test_emb = convert_to_emb(X_test, embedding_dict)

    train_dataset = FastTextDataset(X_train_emb, y_train)
    test_dataset = FastTextDataset(X_test_emb, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = LSTMNet(EMB_DIM, NUM_HIDDEN_NODES, NUM_OUTPUT_NODES, NUM_LAYERS, BIDIRECTION, DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCELoss().to(device)

    print(df.head())

    torch.manual_seed(0)
    num_epochs = 10

    train_loss_hist, test_loss_hist = train(
        model, optimizer, criterion, train_dataloader, test_dataloader, num_epochs, device
    )

    plot_loss(train_loss_hist, test_loss_hist)

if __name__ == "__main__":
    main()
