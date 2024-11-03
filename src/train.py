import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

def train(model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, num_epochs: int, device: str):
    train_loss_hist = []
    test_loss_hist = []
    train_acc_hist = []
    test_acc_hist = []

    with open('logs/training_logs.txt', 'w') as log_file:
        log_file.write("Epoch, Train Loss, Test Loss, Train Accuracy, Test Accuracy\n")
        
        for epoch in range(num_epochs):
            train_losses_epoch = []
            test_losses_epoch = []
            train_acc_epoch = []
            test_acc_epoch = []

            model.train()
            for X_batch, y_batch, text_lengths in tqdm(train_dataloader):
                optimizer.zero_grad()
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output_train = model(X_batch, text_lengths).squeeze()
                loss_train = criterion(output_train, y_batch)
                loss_train.backward()
                train_acc_epoch.append(binary_accuracy(output_train, y_batch).item())
                nn.utils.clip_grad_norm_(model.parameters(), 3)
                optimizer.step()
                train_losses_epoch.append(loss_train.item())
            
            model.eval()
            with torch.no_grad():
                for X_batch_valid, y_batch_valid, val_text_lengths in test_dataloader:
                    X_batch_valid, y_batch_valid = X_batch_valid.to(device), y_batch_valid.to(device)
                    output_test = model(X_batch_valid, val_text_lengths).squeeze()
                    loss_test = criterion(output_test, y_batch_valid)
                    acc_test = binary_accuracy(output_test, y_batch_valid)
                    test_losses_epoch.append(loss_test.item())
                    test_acc_epoch.append(acc_test.item())
                    
            train_loss = np.mean(train_losses_epoch)
            test_loss = np.mean(test_losses_epoch)
            train_acc = np.mean(train_acc_epoch)
            test_acc = np.mean(test_acc_epoch)

            train_loss_hist.append(train_loss)
            test_loss_hist.append(test_loss)

            log_file.write(f"{epoch+1}, {train_loss:.4f}, {test_loss:.4f}, {train_acc:.4f}, {test_acc:.4f}\n")

            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
                print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")

    return train_loss_hist, test_loss_hist
