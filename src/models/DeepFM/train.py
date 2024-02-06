#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.DeepFM.eval import MyEvaluator
from models.DeepFM.model import DeepFM
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm

class MyTrainer:
    def __init__(self, device, num_users, num_items, feature_sizes):
        self.device = device
        self.num_users = num_users
        self.num_items = num_items
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes

    def train_with_hyper_param(self, train_data, hyper_param, verbose=False):
        # Hyperparameters
        epochs = hyper_param['epochs']
        batch_size = hyper_param['batch_size']
        embedding_dim = hyper_param['embedding_dim']
        hidden_dim = hyper_param['hidden_dim']
        learning_rate = hyper_param['learning_rate']

        # Loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        total_batches = len(train_loader)
        model = DeepFM(self.num_users, self.num_items, self.feature_sizes, embedding_dim, hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        pbar = tqdm(range(epochs), leave=False, colour='green', desc='epoch')

        for epoch in pbar:
            avg_loss = 0
            for user, item, features, rating in tqdm(train_loader, leave=False, colour='red', desc='batch'):
                # send data to a running device (GPU or CPU)
                user = user.to(self.device)
                item = item.to(self.device)
                features = features.to(self.device)
                rating = rating.to(self.device)

                optimizer.zero_grad()
                
                predictions = model(user, item, features).squeeze()
                loss = F.mse_loss(predictions, rating)
                rmse = torch.sqrt(loss)
                
                loss.backward()
                optimizer.step()

                avg_loss += loss / total_batches

            if verbose:
                pbar.write('Epoch {:02}: {:.4} training RMSE'.format(epoch, rmse.item()))

        pbar.close()

        return model
