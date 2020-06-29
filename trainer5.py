import os
from math import sqrt

import joblib
import keras
import pandas as pd
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Lambda, Input, LSTM, concatenate, Reshape, Conv1D
from datetime import datetime

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

x, y = list(), list()
for i in range(24, dataset.shape[0] - 23):
    temp = list(dataset[i - 24: i, ].flatten())
    for j in range(0, 24):
        for l in range(1, 12):
            temp.append(dataset[i + j, l])
    x.append(np.array(temp))
    y.append(list(dataset[i: i + 24, 0].flatten()))

x, y = np.array(x), np.array(y)

scaler = MinMaxScaler(feature_range=(0, 1))
x = scaler.fit_transform(x)

x = x.reshape((x.shape[0], 1, 24 * 12 + 11 * 24))
y = y.reshape((y.shape[0], 1, 24))

parallel = 72

if parallel > 1:
    x = np.concatenate([x for i in range(0, parallel)], axis=1)

x = x[12 + 6 + 30:, ]
y = y[12 + 6 + 30:, ]

train_x, train_y = x[:200 * 72, ], y[:200 * 72, ]
test_x, test_y = x[200 * 72:, ], y[200 * 72:, ]

kmodel_in = Input(batch_shape=(24, train_x.shape[1], train_x.shape[2]))


def create_model(index):
    kmodel1 = Lambda(lambda input_x: input_x[:, index: index + 1, 0: 24 * 12], output_shape=(1, 24 * 12))(kmodel_in)
    kmodel2 = Lambda(lambda input_x: input_x[:, index: index + 1, 24 * 12: 24 * 12 + 24 * 11], output_shape=(1, 24 * 11))(kmodel_in)

    kmodel1 = LSTM(50)(kmodel1)
    kmodel1 = Dense(24, activation='relu')(kmodel1)
    kmodel1 = Reshape((24, 1))(kmodel1)

    kmodel2 = Reshape((24, 11))(kmodel2)

    kmodel = concatenate([kmodel1, kmodel2], axis=2)
    kmodel = Conv1D(filters=12, kernel_size=1)(kmodel)
    kmodel = Conv1D(filters=3, kernel_size=1)(kmodel)
    kmodel = Conv1D(filters=1, kernel_size=1)(kmodel)
    kmodel = Reshape((1, 24))(kmodel)
    kmodel = Dense(24, activation='relu')(kmodel)
    kmodel = Dense(24, activation='relu')(kmodel)

    return kmodel


def create_final_model():
    if parallel > 1:
        kmodel = concatenate([create_model(i) for i in range(0, parallel)])
        kmodel = Dense(24, kernel_regularizer=l2(0.01), activation='relu')(kmodel)
    else:
        kmodel = Dense(24, kernel_regularizer=l2(0.01), activation='relu')(create_model(0))
    kmodel = Model(inputs=[kmodel_in], outputs=[kmodel])
    kmodel.compile(optimizer=Adam(), loss='mse', metrics=['acc', 'mape'])
    return kmodel


if __name__ == '__main__':

    test_x = test_x[1:, ]
    test_y = test_y[1:, ]

    if os.path.exists('Model.h5'):
        model = keras.models.load_model('Model.h5')
    else:
        model = create_final_model()
        ckpt = ModelCheckpoint(monitor='val_loss', verbose=1, save_best_only=True, filepath='Model.h5')
        model.fit(train_x, train_y, validation_data=[test_x, test_y], batch_size=24, epochs=500, verbose=1, shuffle=True, callbacks=[ckpt])

    # if not os.path.exists('Model.pkl'):
    #     model = KerasRegressor(build_fn=create_final_model, verbose=1)
    #     batch_size = [10, 20, 40, 60, 80, 100, 200]
    #     epochs = [10, 50, 100, 200]
    #     param_grid = dict(batch_size=batch_size, epochs=epochs)
    #     grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
    #     print('new model created')
    # else:
    #     grid = joblib.load('Model.pkl')
    #     print('model loaded')
    #
    # grid_result = grid.fit(train_x, train_y, validation_data=[test_x, test_y])
    # print(grid_result)
    # joblib.dump(grid, 'Model.pkl')

    yhat = model.predict(test_x, batch_size=24)
    # for error in keras.losses.mean_absolute_percentage_error(test_y, yhat):
    #     print(error)

    print(yhat[yhat.shape[0] - 1, ])
