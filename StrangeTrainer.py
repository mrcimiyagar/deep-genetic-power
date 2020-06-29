import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.metrics import sparse_categorical_accuracy
from numpy import array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
import pandas as pd
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend
import math
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, LinearSVC
import tensorflow.keras as keras

training_batch_size = 24
BATCH_SIZE = 1
n_steps = 24


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        global curr_loss
        self.losses.append(logs.get('loss'))
        curr_loss = logs.get('loss')


def loss(y_true, y_pred):
    return backend.max(backend.abs(y_true - y_pred))


def create_time_series_net():
    time_series_model_in = Input(batch_shape=(training_batch_size, n_steps, 12))
    time_series_model = LSTM(50)(time_series_model_in)
    time_series_model = Dense(24, kernel_initializer='random_uniform', bias_initializer='ones')(time_series_model)

    return time_series_model, time_series_model_in


def create_deep_net():
    deep_net_model_in = Input(batch_shape=(training_batch_size, 24, 14))
    main_model = Dense(24, kernel_initializer='random_uniform', bias_initializer='ones')(deep_net_model_in)
    main_model = Activation(activation="relu")(main_model)
    return main_model, deep_net_model_in


def concat_models():
    ts_model, ts_model_in = create_time_series_net()
    dnn_model, dnn_model_in = create_deep_net()

    main_model = concatenate(inputs=[ts_model, dnn_model])
    main_model = Dense(24, kernel_initializer='random_uniform', bias_initializer='ones')(main_model)
    main_model = Flatten()(main_model)
    main_model = Activation(activation="relu")(main_model)
    main_model = Dense(24, kernel_initializer='random_uniform', bias_initializer='ones')(main_model)
    main_model = Activation(activation="relu")(main_model)
    main_model = Dense(24, kernel_initializer='random_uniform', bias_initializer='ones')(main_model)
    return main_model, ts_model_in, dnn_model_in


def create_main_model():
    ts_model, ts_model_in = create_time_series_net()
    model = Model(inputs=[ts_model_in],
                  outputs=ts_model)
    model.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=['mae', loss, 'mape', 'acc'])
    return model


def make_list_of_additional_data(data):
    result = list()
    for i in range(24, data.shape[0] - n_steps + 1):
        result.append(data[i: i + n_steps, ])
    return result


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(n_steps, len(sequence) - n_steps + 1):
        X.append(sequence[i - n_steps: i, ])
        y.append(sequence[i: i + n_steps, 1])
    return X, y


def predict_final_values(model, scaler1, scaler2):
    def make_list_of_additional_data_for_prediction(data):
        result = list()
        for i in range(24, data.shape[0] - n_steps + 1):
            result.append(data[i: i + n_steps, ])
        return result

    def split_sequence_for_prediction(sequence, n_steps):
        X = list()
        y = list()
        for i in range(n_steps, sequence.shape[0] + 1):
            X.append(sequence[i - n_steps: i])
            if i + n_steps < sequence.shape[0] + 1:
                y.append(list(sequence[i: i + n_steps]))
            else:
                y.append(list([417.51,
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
                               262.14]))
        return X, y

    # model = keras.models.load_model('StrangeModel.h5', custom_objects={ 'loss': loss })

    data = pd.read_excel('dataset1395.xlsx')
    df = pd.DataFrame(data,
                      columns=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H14', 'H14'
                          , 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24'])
    raw = list(array(df))

    data = pd.read_excel('dataset1396.xlsx')
    df = pd.DataFrame(data,
                      columns=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H14', 'H14'
                          , 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24'])
    raw.extend(list(array(df)))

    data = pd.read_excel('dataset1397.xlsx')
    df = pd.DataFrame(data,
                      columns=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H14', 'H14'
                          , 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24'])
    raw.extend(list(array(df)))

    data = pd.read_excel('dataset1398.xlsx')
    df = pd.DataFrame(data,
                      columns=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H14', 'H14'
                          , 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24'])
    raw.extend(list(array(df)))

    raw = array(raw)

    rowCount = raw.shape[0] * raw.shape[1]
    raw = raw.reshape((rowCount))

    data = pd.read_excel('datasetW.xlsx')
    df = pd.DataFrame(data,
                      columns=['year', 'dayOfYear', 'month', 'dayOfWeek', 'hour', 'holiday', 'hourChange', 'specialDay',
                               'Temperature', 'Humidity', 'WeatherState'])

    additionalData = array(df)

    timeSeries = raw

    for i in range(0, timeSeries.shape[0]):
        timeSeries[i] = int(timeSeries[i])

    additionalDataCount = additionalData.shape[0]
    additionalData = additionalData.reshape((additionalDataCount, 11))

    for i in range(len(timeSeries)):
        if math.isnan(timeSeries[i]):
            if 0 < i < len(timeSeries) - 1:
                timeSeries[i] = (timeSeries[i - 1] + timeSeries[i + 1]) / 2
            elif i == 0:
                timeSeries[i] = timeSeries[i + 1]
            elif i == len(timeSeries) - 1:
                timeSeries[i] = timeSeries[i - 1]

    for i in range(additionalData.shape[0]):
        for j in range(additionalData.shape[1]):
            if math.isnan(additionalData[i, j]):
                print(str(i) + ' ' + str(j))
                additionalData[i, j] = 0

    timeSeries = timeSeries[np.logical_not(np.isnan(timeSeries))]

    X, y = split_sequence_for_prediction(timeSeries, n_steps)

    X = array(X)
    y = array(y)

    trainX1 = []
    evalX1 = []
    testX1 = []

    trainX1[:] = np.array(X[0: 18000, ])
    evalX1[:] = np.array(X[18000: 22800, ])
    testX1[:] = np.array(X[22800:, ])

    trainX1 = array(trainX1)
    evalX1 = array(evalX1)
    testX1 = array(testX1)

    additionalData = array(make_list_of_additional_data_for_prediction(additionalData))

    additionalData.resize((additionalData.shape[0], additionalData.shape[1], 14))
    for i in range(0, additionalData.shape[0]):
        for j in range(0, additionalData.shape[1]):
            if i < 24:
                additionalData[i, j, 11] = y[i, j]
                additionalData[i, j, 12] = y[i, j]
                additionalData[i, j, 13] = y[i, j]
                continue
            elif i < 7 * 24:
                additionalData[i, j, 11] = y[i - 24, j]
                additionalData[i, j, 12] = y[i, j]
                additionalData[i, j, 13] = y[i, j]
                continue
            elif i < 365 * 24:
                additionalData[i, j, 11] = y[i - 24, j]
                additionalData[i, j, 12] = y[i - 7 * 24, j]
                additionalData[i, j, 13] = y[i, j]
                continue
            additionalData[i, j, 11] = y[i - 24, j]
            additionalData[i, j, 12] = y[i - 7 * 24, j]
            additionalData[i, j, 13] = y[i - 365 * 24, j]

    trainX2 = np.array(additionalData[0: 18000, ])
    evalX2 = np.array(additionalData[18000: 22800, ])
    testX2 = np.array(additionalData[22800:, ])

    test_output = np.array(y[0: 18000, ])
    eval_output = np.array(y[18000: 22800, ])
    test_output = np.array(y[22800:, ])

    trainX1 = trainX1.reshape((trainX1.shape[0], BATCH_SIZE, 24, 1))
    evalX1 = evalX1.reshape((evalX1.shape[0], BATCH_SIZE, 24, 1))
    testX1 = testX1.reshape((testX1.shape[0], BATCH_SIZE, 24, 1))

    train_input = [trainX1, trainX2, trainX1, trainX2, trainX1, trainX2]
    eval_input = [evalX1, evalX2, evalX1, evalX2, evalX1, evalX2]
    test_input = [testX1, testX2, testX1, testX2, testX1, testX2]

    testInput1 = test_input[0]
    testInput1 = testInput1.reshape((testInput1.shape[0], BATCH_SIZE * n_steps))
    testInput1 = scaler1.transform(testInput1)
    testInput1 = testInput1.reshape((testInput1.shape[0], BATCH_SIZE, n_steps, 1))

    testInput2 = test_input[1]
    testInput2 = testInput2.reshape((testInput2.shape[0], n_steps * 14))
    testInput2 = scaler2.transform(testInput2)
    testInput2 = testInput2.reshape((testInput2.shape[0], n_steps, 14))

    test_input = [testInput1, testInput2, testInput1, testInput2, testInput1, testInput2]

    test_input = [test_input[0][5905 - 48: 5905 - 24, ],
                  test_input[1][5905 - 48: 5905 - 24, ],
                  test_input[0][5905 - 48: 5905 - 24, ],
                  test_input[1][5905 - 48: 5905 - 24, ],
                  test_input[0][5905 - 48: 5905 - 24, ],
                  test_input[1][5905 - 48: 5905 - 24, ]]

    y_pred = model.predict(test_input)
    print(y_pred)

    model.evaluate(test_input, test_output[5905 - 48: 5905 - 24, ])


if __name__ == '__main__':
    data = pd.read_excel('dataset1395.xlsx')
    df = pd.DataFrame(data,
                      columns=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H14', 'H14'
                          , 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24'])
    raw = list(array(df))

    data = pd.read_excel('dataset1396.xlsx')
    df = pd.DataFrame(data,
                      columns=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H14', 'H14'
                          , 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24'])
    raw.extend(list(array(df)))

    data = pd.read_excel('dataset1397.xlsx')
    df = pd.DataFrame(data,
                      columns=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H14', 'H14'
                          , 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24'])
    raw.extend(list(array(df)))

    data = pd.read_excel('dataset1398.xlsx')
    df = pd.DataFrame(data,
                      columns=['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H14', 'H14'
                          , 'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24'])
    raw.extend(list(array(df)))

    raw = array(raw)

    rowCount = raw.shape[0] * raw.shape[1]
    raw = raw.reshape((rowCount))

    data = pd.read_excel('datasetW.xlsx')
    df = pd.DataFrame(data,
                      columns=['year', 'dayOfYear', 'month', 'dayOfWeek', 'hour', 'holiday', 'hourChange', 'specialDay',
                               'Temperature', 'Humidity', 'WeatherState'])

    additionalData = array(df)
    timeSeries = raw

    for i in range(len(timeSeries)):
        if math.isnan(timeSeries[i]):
            if 0 < i < len(timeSeries) - 1:
                timeSeries[i] = (timeSeries[i - 1] + timeSeries[i + 1]) / 2
            elif i == 0:
                timeSeries[i] = timeSeries[i + 1]
            elif i == len(timeSeries) - 1:
                timeSeries[i] = timeSeries[i - 1]

    for i in range(additionalData.shape[0]):
        for j in range(additionalData.shape[1]):
            if math.isnan(additionalData[i, j]):
                print(str(i) + ' ' + str(j))
                additionalData[i, j] = 0

    timeSeries = timeSeries[np.logical_not(np.isnan(timeSeries))]

    additionalData = additionalData[:additionalData.shape[0] - n_steps * BATCH_SIZE, ]
    additionalData = array(additionalData)
    additionalData.resize((additionalData.shape[0], 12))
    additionalData[:, 11] = timeSeries[:]
    temp = []
    temp[:] = additionalData[:, 11]
    additionalData[:, 11] = additionalData[:, 1]
    additionalData[:, 1] = temp[:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(additionalData)
    additionalData = scaler.transform(additionalData)

    X, y = split_sequence(additionalData, n_steps)

    X = array(X)
    y = array(y)

    trainX1 = []
    evalX1 = []
    testX1 = []

    trainX1[:] = np.array(X[0: 18000, ])
    evalX1[:] = np.array(X[18000: 22800, ])
    testX1[:] = np.array(X[22800:, ])

    trainX1 = array(trainX1)
    evalX1 = array(evalX1)
    testX1 = array(testX1)

    additionalData = array(make_list_of_additional_data(additionalData))

    trainX2 = np.array(additionalData[0: 18000, ])
    evalX2 = np.array(additionalData[18000: 22800, ])
    testX2 = np.array(additionalData[22800:, ])

    trainY1 = np.array(y[0: 18000, ])
    evalY1 = np.array(y[18000: 22800])
    testY1 = np.array(y[22800:])

    trainX1 = trainX1.reshape((trainX1.shape[0], 24, 12))
    evalX1 = evalX1.reshape((evalX1.shape[0], 24, 12))
    testX1 = testX1.reshape((testX1.shape[0], 24, 12))

    trainY1 = trainY1.reshape((trainY1.shape[0], 24))
    evalY1 = evalY1.reshape((evalY1.shape[0], 24))
    testY1 = testY1.reshape((testY1.shape[0], 24))

    train_input = [trainX1, trainX2, trainX1, trainX2, trainX1, trainX2]
    eval_input = [evalX1, evalX2, evalX1, evalX2, evalX1, evalX2]
    test_input = [testX1, testX2, testX1, testX2, testX1, testX2]

    train_output = trainY1
    eval_output = evalY1
    test_output = testY1

    if not os.path.exists('StrangeModel.h5'):
        model = create_main_model()
    else:
        model = keras.models.load_model('StrangeModel.h5', custom_objects={'loss': loss})

    model.fit(train_input, train_output, validation_data=[eval_input, eval_output],
              epochs=64, batch_size=training_batch_size, verbose=1, shuffle=False)
    model.save('StrangeModel.h5')
    model.evaluate(test_input, test_output)

    test_input = [test_input[0][test_input[0].shape[0] - 24: test_input[0].shape[0], ],
                  test_input[1][test_input[1].shape[0] - 24: test_input[1].shape[0], ],
                  test_input[0][test_input[0].shape[0] - 24: test_input[0].shape[0], ],
                  test_input[1][test_input[1].shape[0] - 24: test_input[1].shape[0], ],
                  test_input[0][test_input[0].shape[0] - 24: test_input[0].shape[0], ],
                  test_input[1][test_input[1].shape[0] - 24: test_input[1].shape[0], ]]

    # y_pred = model.predict(test_input)
    # print(test_output[test_output.shape[0] - 24: test_output.shape[0]])
    # print(y_pred)
