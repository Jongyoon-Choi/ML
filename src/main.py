#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import fire
import torch
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from utils import preprocessing_data, feature_engineering, extract_numbers, label_encoding
from data import MyDataset
from models.DeepFM.train import MyTrainer
from models.DeepFM.eval import MyEvaluator
from loguru import logger


def run_mymodel(device, train_data, test_data, evaluation_data, hyper_param):
    trainer = MyTrainer(device=device,
                        num_users=train_data.num_users+ test_data.num_users,
                        num_items=train_data.num_items+ test_data.num_items,
                        feature_sizes = [a + b for a, b in zip(train_data.feature_sizes, test_data.feature_sizes)])
    model = trainer.train_with_hyper_param(train_data=train_data,
                                             hyper_param=hyper_param,
                                             verbose=True)
    evaluator = MyEvaluator(device)
    rmse = evaluator.evaluate(model, test_data)

    predictions=evaluator.predict(model, evaluation_data)
    print(predictions[:20])
    

    # 예측 결과를 DataFrame에 추가
    id_column = ['TEST_{:06}'.format(i) for i in range(len(predictions))]

    # 데이터프레임 생성
    result_df = pd.DataFrame({'ID': id_column, 'Book-Rating': predictions})
    # CSV 파일로 저장
    result_df.to_csv('./datasets/BookRating/DeepFM_epoch5.csv', index=False)
    return rmse


def main(model='DeepFM',
         batch_size=64,
         epochs=5,
         embedding_dim = 64, 
         hidden_dim = 64, 
         learning_rate=0.001):
    """
    Handle user arguments of ml-project-template

    :param model: name of model to be trained and tested
    :param seed: random_seed (if -1, a default seed is used)
    :param batch_size: size of batch
    :param epochs: number of training epochs
    :param learning_rate: learning rate
    """

    # Step 0. Initialization
    logger.info("The main procedure has started with the following parameters:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    param = dict()
    param['model'] = model
    param['device'] = device

    # Step 1. Load datasets
    folder_path = Path(__file__).parents[1].absolute().joinpath("datasets", "BookRating")
    data_path = folder_path.joinpath("train.csv")
    evaluation_path = folder_path.joinpath("test.csv")
    data = pd.read_csv(data_path, encoding='utf-8')
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    evaluation_data = pd.read_csv(evaluation_path, encoding='utf-8')
   
    # data 전처리
    train_data = preprocessing_data(train_data)
    train_data = feature_engineering(train_data)
    train_data = extract_numbers(train_data)
    
    test_data = preprocessing_data(test_data)
    test_data = feature_engineering(test_data)
    test_data = extract_numbers(test_data)
    
    evaluation_data = preprocessing_data(evaluation_data)
    evaluation_data = feature_engineering(evaluation_data)
    evaluation_data = extract_numbers(evaluation_data)
    train_data, test_data, evaluation_data = label_encoding(train_data, test_data, evaluation_data)

    train_dataset = MyDataset(train_data)
    test_dataset = MyDataset(test_data)
    evaluation_dataset = MyDataset(evaluation_data)
    print(evaluation_dataset.data)


    logger.info("The datasets are loaded where their statistics are as follows:")
    logger.info("- # of training instances: {}".format(len(train_dataset)))
    logger.info("- # of test instances: {}".format(len(test_dataset)))

    # Step 2. Run (train and evaluate) the specified model

    logger.info("Training the model has begun with the following hyperparameters:")
    hyper_param = dict()
    hyper_param['batch_size'] = batch_size
    hyper_param['epochs'] = epochs
    hyper_param['embedding_dim'] = embedding_dim
    hyper_param['hidden_dim'] = hidden_dim
    hyper_param['learning_rate'] = learning_rate

    if model == 'DeepFM':
        rmse = run_mymodel(device=device,
                               train_data = train_dataset,
                               test_data = test_dataset, 
                               evaluation_data = evaluation_dataset,
                               hyper_param=hyper_param)

        # - If you want to add other model, then add an 'elif' statement with a new runnable function
        #   such as 'run_my_model' to the below
        # - If models' hyperparamters are varied, need to implement a function loading a configuration file
    else:
        logger.error("The given \"{}\" is not supported...".format(model))
        return

    # Step 3. Report and save the final results
    logger.info("The model has been trained. The test accuracy is {:.4}.".format(rmse))


if __name__ == "__main__":
    sys.exit(fire.Fire(main))
