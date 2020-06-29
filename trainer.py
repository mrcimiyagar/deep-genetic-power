import os

import keras
import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import LSTM, Dense, Flatten, Reshape, Activation
from keras.optimizers import RMSprop
from keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
additionalData[:, 11] = timeSeries[:]
additionalData = np.array(additionalData)

scaler = MinMaxScaler(feature_range=(0, 1))
additionalData = scaler.fit_transform(additionalData)

x, y = list(), list()
for i in range(48, additionalData.shape[0]):
    x.append(additionalData[i - 48: i, ])
    y.append(additionalData[i, ])

x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)

train_x, train_y = x[0: 200 * 72, ], y[0: 200 * 72, ]
test_x, test_y = x[198 * 72 + 24:, ], y[198 * 72 + 24:, ]

model = Sequential()
model.add(LSTM(50, stateful=True, return_sequences=True, batch_input_shape=(72, x.shape[1], x.shape[2])))
model.add(LSTM(10, stateful=True))
model.add(Dense(12, activation="relu", kernel_regularizer=l2(0.01)))
model.compile(optimizer='adam', loss='mae', metrics=['acc'])

if os.path.exists('Model.h5'):
    model = keras.models.load_model('Model.h5')

model.fit(x=train_x, y=train_y, validation_data=[test_x, test_y], batch_size=72, epochs=50, verbose=1, shuffle=False)
model.evaluate(x=test_x, y=test_y, batch_size=72, verbose=1)
model.save('Model.h5')

yhat = model.predict(test_x, batch_size=72)

test_y = test_y.reshape((test_y.shape[0], 1, 12))
yhat = yhat.reshape(yhat.shape[0], 1, 12)
temp_test_x = np.zeros((test_y.shape[0], test_y.shape[1], 12))
temp_test_x[:, :, :] = test_y[:, :, :]
yhat = yhat.reshape((yhat.shape[0], 1, 12))
temp_test_x[:, :, 11] = yhat[:, :, 11]
temp_test_x = temp_test_x.reshape((temp_test_x.shape[0], temp_test_x.shape[1] * temp_test_x.shape[2]))
inv_yhat = scaler.inverse_transform(temp_test_x)
inv_yhat = inv_yhat.reshape((inv_yhat.shape[0], 1, 12))

for i in range(inv_yhat.shape[0] - 24, inv_yhat.shape[0] - 0):
    print(inv_yhat[i, ])


