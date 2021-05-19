#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:47:29 2021

@author: leo
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook as tqdm

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import RNNModel, ExponentialSmoothing
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

#Read data:
df = pd.read_csv('AirPassengers.csv', delimiter=",")
series = TimeSeries.from_dataframe(df, 'Month', ['Passengers'])

#Create training and validation sets:
train, val = series.split_after(pd.Timestamp('19590101'))

#Normalize the time series (note: we avoid filtering the transformer on the validation set)
transformer = Scaler()
train_transformed = transformer.fit_transform(train)
val_transformed = transformer.transform(val)
series_transformed = transformer.transform(series)

my_model = RNNModel(
    model = 'LSTM',
    input_chunk_length=12,
    output_chunk_length=1,
    hidden_size=25,
    n_rnn_layers=1,
    dropout=0.4,
    batch_size=16,
    n_epochs=400,
    optimizer_kwargs={'lr':1e-3},
    model_name='Air_RNN',
    log_tensorboard=True,
    random_state=42
)

my_model.fit(train_transformed, val_series=val_transformed, verbose=True)

def eval_model(model):
    pred_series = model.predict(n=26)
    plt.figure(figsize=(8,5))
    series_transformed.plot(label='actual')
    pred_series.plot(label='forecast')
    plt.title('MAPE: {:.2f}%'.format(mape(pred_series, val_transformed)))
    plt.legend()
    
eval_model(my_model)

best_model = RNNModel.load_from_checkpoint(model_name='Air_RNN', best=True)
eval_model(best_model)

backtest_series = my_model.historical_forecasts(series_transformed,
                                                start=pd.Timestamp('19590101'),
                                                forecast_horizon=6,
                                                retrain=False,
                                                verbose=True)

plt.figure(figsize=(8,5))
series_transformed.plot(label='actual')
backtest_series.plot(label='backtest')
plt.legend()
plt.title('Backtest, starting Jan 1959, 6-months horizon')
print('MAPE: {:.2f}%'.format(mape(transformer.inverse_transform(series_transformed),
                                  transformer.inverse_transform(backtest_series))))