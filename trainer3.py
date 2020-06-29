from math import sqrt

import pandas as pd
from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from datetime import datetime

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

data = pd.read_excel('datasetW.xlsx')
df = pd.DataFrame(data,
                  columns=['year', 'dayOfYear', 'month', 'dayOfWeek', 'hour', 'holiday', 'hourChange', 'specialDay',
                           'Temperature', 'Humidity', 'WeatherState'])

additionalData = np.array(df)

timeSeries = np.array(timeSeries)
additionalData = additionalData[0: additionalData.shape[0] - 24, ]
additionalData = np.array(additionalData)
additionalData.resize((additionalData.shape[0], 12))
temp = additionalData[:, 0]
additionalData[:, 0] = timeSeries[:]
additionalData[:, 11] = temp[:]
additionalData = np.array(additionalData)
dataset = additionalData


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# ensure all data is float
dataset = dataset.astype('float32')
# frame as supervised learning
reframed = series_to_supervised(dataset, 24, 24)

droppingColumns = []
for i in range(0, 24):
    for j in range(0, 7):
        droppingColumns.append(193 + i * 8 + j)

reframed.drop(reframed.columns[droppingColumns], axis=1, inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
reframed = scaler.fit_transform(reframed)

reframed = DataFrame(reframed)

# split into train and test sets
values = reframed.values

n_train_hours = 200 * 72
train = values[:n_train_hours, :]
test = values[n_train_hours + 25:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# design network
model = Sequential()
model.add(LSTM(50, stateful=True, return_sequences=True, batch_input_shape=(72, train_X.shape[1], train_X.shape[2])))
model.add(LSTM(10, stateful=True))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X, batch_size=72)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast

print(yhat.shape)
print(test_X.shape)

inv_yhat = concatenate((yhat, test_X[:, :]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, :]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

print(inv_y)
print()
print(inv_yhat)
