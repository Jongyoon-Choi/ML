#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pandas as pd
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe.iloc[:, 1:]
        self.num_users = len(self.data['User-ID'].unique())
        self.num_items = len(self.data['Book-ID'].unique())
        features = ['Book-Author', 'Year-Of-Publication', 'Publisher', 'Main_Title', 'Sub_Title', 'City', 'State', 'Country', 'Age_gb']
        self.feature_sizes = [len(self.data[col].unique()) for col in features]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # PyTorch 모델에 입력할 수 있도록 데이터 반환
        user = torch.tensor(self.data.iloc[idx]['User-ID'], dtype=torch.int)
        item = torch.tensor(self.data.iloc[idx]['Book-ID'], dtype=torch.int)
        features = torch.tensor(self.data.iloc[idx, ~self.data.columns.isin(['User-ID', 'Book-ID', 'Book-Rating'])].values, dtype=torch.int)
        rating = torch.tensor(self.data.iloc[idx]['Book-Rating'] if 'Book-Rating' in self.data.columns else 0, dtype=torch.float)

        return user, item, features, rating
