#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class MyEvaluator:
    def __init__(self, device):
        self.device = device

    def evaluate(self, model, test_data):
        model.eval()
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)
    
        all_ratings = []
        all_predictions = []

        with torch.no_grad():
             for user, item, features, rating in tqdm(test_loader, desc='평가 중', leave=False):
                # GPU 또는 CPU로 데이터 이동
                user, item, features, rating = user.to(self.device), item.to(self.device), features.to(self.device), rating.to(self.device)

                # 모델 예측
                output = model(user, item, features).squeeze()

                # 결과 기록
                all_ratings.extend(rating.cpu().numpy())
                all_predictions.extend(output.cpu().numpy())

        # 평균 제곱근 오차 (RMSE) 계산
        rmse = np.sqrt(mean_squared_error(all_ratings, all_predictions))

        print(f'평균 제곱 오차 (RMSE): {rmse:.4f}')

        return rmse
    def predict(self, model, evaluation_data):
        model.eval()
        evaluation_loader = torch.utils.data.DataLoader(evaluation_data, batch_size=64, shuffle=False)
        all_predictions = []
        # 훈련된 모델을 사용하여 예측
        with torch.no_grad():
            for user, item, features, _ in tqdm(evaluation_loader, desc='예측 중', leave=False):
                # GPU 또는 CPU로 데이터 이동
                user, item, features= user.to(self.device), item.to(self.device), features.to(self.device)
                # 모델 예측
                output = model(user, item, features).squeeze()

                # 결과 기록
                all_predictions.extend(output.cpu().numpy())

        return all_predictions