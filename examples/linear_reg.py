import numpy as np
import torch
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

def load_data():
    data = fetch_california_housing() 
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, y_train), (X_test, y_test)

def prepare_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = torch.tensor(X_scaled, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return X, y

class LinearReg:
    def __init__(self, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = nn.Linear(8, 1, device=device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001) 

    def eval(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            y_test = y_test.to(self.device)
            y_pred = self.model(X_test)
            mse = self.criterion(y_pred, y_test)
            print(f'Test MSE: {mse.item():.4f}')
        return mse.item()

    def train(self, X, y, epochs=100, batch_size=64):
        dataset = TensorDataset(X.to(self.device), y.to(self.device))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for X_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}') 
        
        


    

     
