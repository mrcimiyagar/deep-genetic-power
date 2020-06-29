import os
import sys

import keras
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense, LSTM, Reshape
from matplotlib import pyplot
from pandas import concat, DataFrame
from sklearn.preprocessing import MinMaxScaler


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    agg = concat(cols, axis=1)
    agg.dropna(inplace=True)
    print(agg)
    return agg


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
raw = raw.reshape(rowCount)

data = pd.read_excel('datasetW.xlsx')
df = pd.DataFrame(data,
                  columns=['year', 'dayOfYear', 'month', 'dayOfWeek', 'hour', 'holiday', 'hourChange', 'specialDay',
                           'Temperature', 'Humidity', 'WeatherState'])

additionalData = np.array(df)

timeSeries = list(raw)
timeSeries = np.array(timeSeries)

additionalData.resize((additionalData.shape[0], 12))

additionalData = additionalData[: additionalData.shape[0] - 24, ]

additionalData[:, 11] = timeSeries[:]

temp = []
temp[:] = additionalData[:, 11]
additionalData[:, 11] = additionalData[:, 1]
additionalData[:, 1] = temp[:]

values = additionalData
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(values, 24, 24)

# split into train and test sets
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :train.shape[1] - 12 * 24], train[:, train.shape[1] - 12 * 24:]
test_X, test_y = test[:, :test.shape[1] - 12 * 24], test[:, test.shape[1] - 12 * 24:]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print("\n")
print(train_X[0, ])
print("\n")
print(train_y[0, ])
print("\n")
print(test_X[0, ])
print("\n")
print(test_y[0, ])

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(288))
model.compile(loss='mae', optimizer='adam')

if os.path.exists('Model.h5'):
    model = keras.models.load_model('Model.h5')
else:
    history = model.fit(train_X, train_y, epochs=50, batch_size=24, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    model.save('Model.h5')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

yhat = model.predict(test_X)
test_X = test_X[:, :, 1:].reshape((test_X.shape[0], 24 * 12))
inv_yhat = np.concatenate((yhat, test_X), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)