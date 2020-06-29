import os
from math import sqrt

import keras
import pandas as pd
from keras import Input, Model
from keras.optimizers import Adam, SGD
from keras.regularizers import l2, l1_l2
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, concatenate, Reshape, Conv1D, Flatten
from keras.layers import LSTM
from datetime import datetime
import tensorflow as tf

from tensorflow_core.python.keras.optimizer_v2.rmsprop import RMSProp

data = pd.read_excel('dataset1395.xlsx')
df = pd.DataFrame(data,
                  columns=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H14', 'H14'
                      , 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24'])
raw = list(np.array(df))

data = pd.read_excel('dataset1396.xlsx')
df = pd.DataFrame(data,
                  columns=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H14', 'H14'
                      , 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24'])
raw.extend(list(np.array(df)))

data = pd.read_excel('dataset1397.xlsx')
df = pd.DataFrame(data,
                  columns=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H14', 'H14'
                      , 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24'])
raw.extend(list(np.array(df)))

data = pd.read_excel('dataset1398.xlsx')
df = pd.DataFrame(data,
                  columns=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H14', 'H14'
                      , 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24'])
raw.extend(list(np.array(df)))

raw = np.array(raw)
rowCount = raw.shape[0] * raw.shape[1]
timeSeries = list(raw.reshape(rowCount))

timeSeries.extend([
    417.51,
    394.82,
    359.13,
    329.58,
    303.47,
    275.44,
    244.91,
    238.28,
    252.34,
    278.81,
    309.94,
    326.57,
    344.83,
    364.66,
    358.95,
    313.13,
    274.05,
    245.75,
    237.56,
    236.47,
    246.38,
    263.63,
    266.43,
    262.14,
])

# timeSeries = [int(timeSeries[i] / 10) for i in range(0, len(timeSeries))]

print(timeSeries)

data = pd.read_excel('datasetW.xlsx')
df = pd.DataFrame(data,
                  columns=['year', 'dayOfYear', 'month', 'dayOfWeek', 'hour', 'holiday', 'hourChange', 'specialDay',
                           'Temperature', 'Humidity', 'WeatherState'])

additionalData = np.array(df)

timeSeries = np.array(timeSeries)
additionalData = additionalData[0: additionalData.shape[0], ]
additionalData = np.array(additionalData)
additionalData.resize((additionalData.shape[0], 12))
temp = []
temp[:] = additionalData[:, 0]
additionalData[:, 0] = timeSeries[:]
additionalData[:, 11] = temp[:]
dataset = np.array(additionalData)
additionalData.resize((additionalData.shape[0], 24, 14))

for i in range(0, additionalData.shape[0]):
    curr_pos = i
    if curr_pos < 24:
        additionalData[i, 11] = timeSeries[curr_pos]
        additionalData[i, 12] = timeSeries[curr_pos]
        additionalData[i, 13] = timeSeries[curr_pos]
        continue
    if curr_pos < 7 * 24:
        additionalData[i, 11] = timeSeries[curr_pos - 24]
        additionalData[i, 12] = timeSeries[curr_pos]
        additionalData[i, 13] = timeSeries[curr_pos]
        continue
    if curr_pos < 365 * 24:
        additionalData[i, 11] = timeSeries[curr_pos - 24]
        additionalData[i, 12] = timeSeries[curr_pos - 24 * 7]
        additionalData[i, 13] = timeSeries[curr_pos]
        continue
    additionalData[i, 11] = timeSeries[curr_pos - 24]
    additionalData[i, 12] = timeSeries[curr_pos - 24 * 7]
    additionalData[i, 13] = timeSeries[curr_pos - 24 * 365]

x1, x2, y = list(), list(), list()
for i in range(24, dataset.shape[0] - 23):
    x1.append(dataset[i - 24: i, ])
    x2.append(dataset[i, 1:])
    y.append(list(dataset[i, 0]))

x1, x2, y = np.array(x1), np.array(x2), np.array(y)

x1 = x1.reshape((x1.shape[0], 24 * 12))
x2 = x2.reshape((x2.shape[0], 1 * 11))
y = y.reshape((y.shape[0], 1, 1))

scaler1 = MinMaxScaler(feature_range=(0, 1))
x1 = scaler1.fit_transform(x1)
scaler2 = MinMaxScaler(feature_range=(0, 1))
x2 = scaler2.fit_transform(x2)

x1 = x1.reshape((x1.shape[0], 1, 24 * 12))
x2 = x2.reshape((x2.shape[0], 1, 11))

train_x1, train_x2, train_y = x1[:365 * 24, ], x2[:365 * 24, ], y[:365 * 24, ]
test_x1, test_x2, test_y = x1[365 * 24:, ], x2[365 * 24:, ], y[365 * 24:, ]

model1_in = Input(batch_shape=(72, train_x1.shape[1], train_x1.shape[2]))
model1 = LSTM(50, kernel_initializer='random_uniform', bias_initializer='zeros')(model1_in)
model1 = Dense(24, kernel_initializer='random_uniform', bias_initializer='zeros')(model1)
model1 = Reshape((1, 24))(model1)

model2_in = Input(batch_shape=(72, train_x2.shape[1], train_x2.shape[2]))

model = concatenate([model1, model2_in], axis=2)
model = Dense(1, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(model)
model = Model(inputs=[model1_in, model2_in], outputs=[model])
model.compile(optimizer=Adam(), loss='mae', metrics=['mae', 'mape', 'acc'])

if os.path.exists('Model.h5'):
    model = keras.models.load_model('Model.h5')

test_x1 = test_x1[1:, ]
test_x2 = test_x2[1:, ]
test_y = test_y[1:, ]

model.fit(x=[train_x1, train_x2], y=train_y, validation_data=[[test_x1, test_x2], test_y], verbose=1, batch_size=72,
          epochs=500, shuffle=True)
print(model.evaluate([test_x1, test_x2], test_y, verbose=1, batch_size=72))
model.save('Model.h5')
