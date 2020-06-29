import os

import keras
import numpy as np
import pandas as pd
from keras.layers import LSTM, Input, Dense, Reshape, Conv1D, Lambda, TimeDistributed, MaxPooling1D, \
    Flatten, GlobalAveragePooling1D, concatenate, Permute
from keras.models import Model
from keras.optimizers import Adam, Nadam
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler

from tester import Tester

window_size = 125
lstm_units = 34
dense_units = 114
batch_size = 36
init_mode = 'he_uniform'
last_layer_activation = 'linear'
middle_layer_activation = 'softplus'
optimizer = Nadam
learn_rate = 0.01


def prepare_data_set():
    data = pd.read_excel('datasetW.xlsx', sheet_name='input')
    df = pd.DataFrame(data,
                      columns=['year', 'dayOfYear', 'month', 'dayOfWeek', 'hour', 'holiday', 'hourChange', 'specialDay',
                               'Temperature', 'Humidity', 'WeatherState'])
    weather = np.array(df)
    data = pd.read_excel('datasetW.xlsx', sheet_name='output')
    df = pd.DataFrame(data, columns=['Load'])
    load = np.array(df)

    weather.resize((weather.shape[0], 12))
    temp = []
    temp[:] = weather[:, 0]
    weather[:, 0] = load[:, 0]
    weather[:, 11] = temp[:]
    dataset = np.array(weather)
    dataset.resize((dataset.shape[0], 14))

    for i in range(0, dataset.shape[0]):
        if i < 24:
            dataset[i, 11] = dataset[i, 0]
            dataset[i, 12] = dataset[i, 0]
            dataset[i, 13] = dataset[i, 0]
            continue
        if i < 24 * 7:
            dataset[i, 11] = dataset[i - 24, 0]
            dataset[i, 12] = dataset[i, 0]
            dataset[i, 13] = dataset[i, 0]
            continue
        if i < 365 * 24:
            dataset[i, 11] = dataset[i - 24, 0]
            dataset[i, 12] = dataset[i - 24 * 7, 0]
            dataset[i, 13] = dataset[i, 0]
            continue
        dataset[i, 11] = dataset[i - 24, 0]
        dataset[i, 12] = dataset[i - 24 * 7, 0]
        dataset[i, 13] = dataset[i - 365 * 24, 0]

    dataset = dataset.astype('float32')

    x, y = list(), list()
    for i in range(window_size, dataset.shape[0] - 23):
        temp = list(dataset[i - window_size: i, ].flatten())
        for j in range(0, 24):
            for l in range(1, 14):
                temp.append(dataset[i + j, l])
        x.append(np.array(temp))
        y.append(list(dataset[i: i + 24, 0].flatten()))

    x, y = np.array(x), np.array(y)

    scaler_minmax = MinMaxScaler(feature_range=(0, 1))
    x = scaler_minmax.fit_transform(x)

    normalizer = Normalizer()
    x = normalizer.fit_transform(x)

    x = x.reshape((x.shape[0], 1, window_size * 14 + 24 * 13))
    y = y.reshape((y.shape[0], 1, 24))

    train_x, train_y, eval_x, eval_y, testX, testY = \
        np.array(x[0: 200 * 72, ]), np.array(y[0: 200 * 72, ]), \
        np.array(x[200 * 72: 300 * 72, ]), np.array(y[200 * 72: 300 * 72, ]), \
        np.array(x[300 * 72:, ]), np.array(y[300 * 72:, ])

    extra = train_x.shape[0] - (train_x.shape[0] // batch_size) * batch_size
    train_x = train_x[extra:, ]
    train_y = train_y[extra:, ]
    extra = eval_x.shape[0] - (eval_x.shape[0] // batch_size) * batch_size
    eval_x = eval_x[extra:, ]
    eval_y = eval_y[extra:, ]
    extra = testX.shape[0] - (testX.shape[0] // batch_size) * batch_size
    testX = testX[extra:, ]
    testY = testY[extra:, ]

    return train_x, train_y, eval_x, eval_y, testX, testY


def prepare_test_set():
    data = pd.read_excel('datasetW.xlsx', sheet_name='input')
    df = pd.DataFrame(data,
                      columns=['year', 'dayOfYear', 'month', 'dayOfWeek', 'hour', 'holiday', 'hourChange', 'specialDay',
                               'Temperature', 'Humidity', 'WeatherState'])
    weather = np.array(df)
    data = pd.read_excel('datasetW.xlsx', sheet_name='output')
    df = pd.DataFrame(data, columns=['Load'])
    load = np.array(df)

    weather.resize((weather.shape[0], 12))
    temp = []
    temp[:] = weather[:, 0]
    weather[:, 0] = load[:, 0]
    weather[:, 11] = temp[:]
    dataset = np.array(weather)
    dataset.resize((dataset.shape[0], 14))

    for i in range(0, dataset.shape[0]):
        if i < 24:
            dataset[i, 11] = dataset[i, 0]
            dataset[i, 12] = dataset[i, 0]
            dataset[i, 13] = dataset[i, 0]
            continue
        if i < 24 * 7:
            dataset[i, 11] = dataset[i - 24, 0]
            dataset[i, 12] = dataset[i, 0]
            dataset[i, 13] = dataset[i, 0]
            continue
        if i < 365 * 24:
            dataset[i, 11] = dataset[i - 24, 0]
            dataset[i, 12] = dataset[i - 24 * 7, 0]
            dataset[i, 13] = dataset[i, 0]
            continue
        dataset[i, 11] = dataset[i - 24, 0]
        dataset[i, 12] = dataset[i - 24 * 7, 0]
        dataset[i, 13] = dataset[i - 365 * 24, 0]

    dataset = dataset.astype('float32')

    x, y = list(), list()
    for i in range(window_size, dataset.shape[0] - 23):
        temp = list(dataset[i - window_size: i, ].flatten())
        for j in range(0, 24):
            for l in range(1, 14):
                temp.append(dataset[i + j, l])
        x.append(np.array(temp))
        y.append(list(dataset[i: i + 24, 0].flatten()))

    x, y = np.array(x), np.array(y)

    scaler_minmax = MinMaxScaler(feature_range=(0, 1))
    x = scaler_minmax.fit_transform(x)

    normalizer = Normalizer()
    x = normalizer.fit_transform(x)

    x = x.reshape((x.shape[0], 1, window_size * 14 + 24 * 13))
    y = y.reshape((y.shape[0], 1, 24))

    testX, testY = np.array(x), np.array(y)

    extra = testX.shape[0] - (testX.shape[0] // batch_size) * batch_size

    testX = testX[extra:, ]
    testY = testY[extra:, ]

    return testX, testY


def create_model():
    global window_size
    global lstm_units
    global dense_units
    global kmodel_in
    global init_mode
    global last_layer_activation
    global middle_layer_activation

    kmodel1 = Lambda(lambda input_x: input_x[:, :, 0: window_size * 14],
                     output_shape=(1, window_size * 14))(kmodel_in)
    kmodel2 = Lambda(lambda input_x: input_x[:, :, window_size * 14: window_size * 14 + 24 * 13],
                     output_shape=(1, 24 * 13))(kmodel_in)

    kmodel3 = Reshape((window_size, 14, 1))(kmodel1)
    kmodel3 = Lambda(lambda input_x: (input_x[:, :, 0: 1, :]), output_shape=(window_size, 1))(kmodel3)
    kmodel3 = Reshape((1, window_size, 1))(kmodel3)

    kmodel1 = LSTM(lstm_units, kernel_initializer=init_mode, activation=middle_layer_activation)(kmodel1)
    kmodel1 = Dense(dense_units, activation=middle_layer_activation, kernel_initializer=init_mode)(kmodel1)
    kmodel1 = Dense(dense_units, activation=middle_layer_activation, kernel_initializer=init_mode)(kmodel1)
    kmodel1 = Dense(dense_units, activation=middle_layer_activation, kernel_initializer=init_mode)(kmodel1)
    kmodel1 = Reshape((1, dense_units))(kmodel1)

    kmodel2_main = Reshape((24, 13))(kmodel2)
    kmodel2 = Lambda(lambda input_x: input_x[:, :, :10], output_shape=(24, 10))(kmodel2_main)
    kmodel2 = Reshape((24, 10))(kmodel2)
    kmodel4 = Lambda(lambda input_x: input_x[:, :, 10:], output_shape=(24, 3))(kmodel2_main)
    kmodel4 = Reshape((24, 3))(kmodel4)

    kmodel2 = Flatten()(kmodel2)
    kmodel2 = Dense(dense_units, activation=middle_layer_activation, kernel_initializer=init_mode)(kmodel2)
    kmodel2 = Dense(dense_units, activation=middle_layer_activation, kernel_initializer=init_mode)(kmodel2)
    kmodel2 = Dense(dense_units, activation=middle_layer_activation, kernel_initializer=init_mode)(kmodel2)
    kmodel2 = Reshape((1, dense_units))(kmodel2)

    kmodel3 = TimeDistributed(Conv1D(filters=5, kernel_size=3, kernel_initializer=init_mode,
                                     activation=middle_layer_activation))(kmodel3)
    kmodel3 = TimeDistributed(MaxPooling1D(pool_size=2))(kmodel3)
    kmodel3 = TimeDistributed(Flatten())(kmodel3)
    kmodel3 = LSTM(lstm_units, stateful=True, return_sequences=True)(kmodel3)
    kmodel3 = LSTM(lstm_units // 5 if lstm_units > 5 else lstm_units, stateful=True)(kmodel3)
    kmodel3 = Dense(dense_units, activation=middle_layer_activation, kernel_initializer=init_mode)(kmodel3)
    kmodel3 = Dense(dense_units, activation=middle_layer_activation, kernel_initializer=init_mode)(kmodel3)
    kmodel3 = Reshape((1, dense_units))(kmodel3)

    kmodel4 = Flatten()(kmodel4)
    kmodel4 = Dense(dense_units, activation=middle_layer_activation, kernel_initializer=init_mode)(kmodel4)
    kmodel4 = Dense(dense_units, activation=middle_layer_activation, kernel_initializer=init_mode)(kmodel4)
    kmodel4 = Reshape((1, dense_units))(kmodel4)

    kmodel = concatenate([kmodel1, kmodel2, kmodel3, kmodel4], axis=1)
    kmodel =
    kmodel = Dense(24, activation=middle_layer_activation, kernel_initializer=init_mode)(kmodel)
    kmodel = Dense(24, activation=last_layer_activation, kernel_initializer=init_mode)(kmodel)

    return kmodel


def create_final_model():
    global optimizer
    global learn_rate

    kmodel = create_model()
    kmodel = Model(inputs=[kmodel_in], outputs=[kmodel])
    kmodel.compile(optimizer=optimizer(learning_rate=learn_rate), loss='mse', metrics=['acc', 'mape'])
    return kmodel


train_x, train_y, eval_x, eval_y, test_x, test_y = prepare_data_set()

kmodel_in = Input(batch_shape=(batch_size, test_x.shape[1], test_x.shape[2]))

model = create_final_model()

model.fit(x=train_x, y=train_y[:, 0, :], batch_size=batch_size, epochs=100, verbose=1,
          validation_data=[eval_x, eval_y[:, 0, :]],
          shuffle=True)
model.save('Final_Model.h5')

tester = Tester('result_without_scaler')
tester.test_neuron(model, test_x, test_y, batch_size)
