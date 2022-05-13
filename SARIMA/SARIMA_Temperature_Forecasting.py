#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import datetime
import timeit

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import optimizers
from pandas.core.frame import DataFrame
from tensorflow.keras.optimizers import Adam
import math
from tqdm import tqdm_notebook
from itertools import product


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

start = timeit.default_timer()


# In[ ]:


from pandas.core.frame import DataFrame
import pandas as pd
import math


class Src:

    def __init__(self, name: str = 'https://raw.githubusercontent.com/isu92neth/Weather-RNN/main/weatherHistory.csv'):
      
        self.name = name
        self.df = DataFrame()

        self.train_df = DataFrame()
        self.test_df = DataFrame()
        self.val_df = DataFrame()

        self.titles = ['Date', 'Summary', 'Precip Type', 'Temperature',
                       'Apparent Temperature', 'Humidity', 'Wind Speed',
                       'Wind Bearing', 'Visibility', 'Loud Cover',
                       'Pressure', 'Daily Summary']

    def load(self) -> DataFrame:
        # load hungarian dataset
        df = pd.read_csv(self.name)
        assert df is not None, 'file error'
        assert len(df) > 0, 'file is empty'
        # set up titles

        df.columns = self.titles

        # index form
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df = df.set_index('Date')

        # sort
        self.df = df.sort_index()

        return self.df

    def remove_Unnecessary_Input(self) -> DataFrame:
        del self.df['Summary']
        del self.df['Precip Type']
        del self.df['Daily Summary']
        del self.df['Loud Cover']

        return self.df

    def average(self, interval: int = 7*24) -> DataFrame:
        # average every 6h
        df = DataFrame(columns=self.df.keys())
        # print(self.df['Temperature'])
        for x in range(math.floor(len(self.df) / interval)):
            # tem =

            tem = self.df[:][x * interval:(x + 1) * interval]
            date = pd.to_datetime(tem.index.values[0])

            average = tem.mean().values.T
            tem_entry = DataFrame([average], columns=self.df.keys(), index=[date])
            tem_frame = [df, tem_entry]
            df = pd.concat(tem_frame)

        self.df = df

        return self.df

   

    def run_Src(self):

        df = self.load()
        self.remove_Unnecessary_Input()
        return self.average()


# In[ ]:


file = Src()
df = file.run_Src()


# In[ ]:


df = df['Temperature']
df
_ = df.plot(subplots=True)
plt.xlabel("Date")
plt.ylabel("Temperature")


# In[ ]:


from statsmodels.tsa.stattools import adfuller
ad_fuller_result = adfuller(df)
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')  


# In[ ]:


X = df.values
size = int(len(X) * 0.9)
train, test = X[0:size], X[size:len(X)]

#normalizing
train_mean = train.mean()
train_std = train.std()

train = (train - train_mean) / train_std
test = (test - train_mean) / train_std

history = [x for x in train]


# In[ ]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[ ]:


from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

pred_exp = []
residuals = []
predictions = list()
summary = []
test = test[:52]
my_order = (1,0,3)
my_seasonal_order = (1, 0, 1, 52)

for t in range(len(test)):
    model = SARIMAX(history, order=my_order, seasonal_order=my_seasonal_order, initialization='approximate_diffuse')
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
    pred_exp.append((yhat, obs))
    residuals.append((obs - yhat))
# evaluate forecasts
print(model_fit.summary())
mse = (mean_squared_error(test, predictions))
print('Test MSE: %.3f' % mse)
# plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()


# In[ ]:


model_fit.plot_diagnostics(figsize=(15,12));

