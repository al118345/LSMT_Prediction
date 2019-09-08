from ConexionBaseDeDatos import ConexionBaseDeDatos
# Import dependencies
import pandas as pd
import numpy as np
import datetime
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import datetime
import datetime
from matplotlib import pyplot


from sklearn.model_selection import train_test_split
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional

from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


if __name__ == '__main__':
    conexionBase = ConexionBaseDeDatos()
    conexionBase.prediccion_Datos_Polaridad()
    resultado={}
    resultado["precio_bitcoin"] = []
    resultado["pol_positivo"] = []
    resultado["pol_negativo"] = []
    resultado["pol_neutral"] = []
    resultado["sentimineto_negativo"] = []
    resultado["sentimiento_positivo"] = []
    resultado["total"] = []
    resultado["fecha"] = []



    for i in conexionBase.prediccion_Datos_Polaridad():
        resultado["precio_bitcoin"].append(float(i[1].replace(",",".")))
        resultado["pol_positivo"].append(float(i[2]))
        resultado["pol_negativo"].append(float(i[3]))
        resultado["pol_neutral"].append(float(i[4]))
        resultado["sentimineto_negativo"].append(i[5])
        resultado["sentimiento_positivo"].append(i[6])
        resultado["total"].append(float(i[7]))
        resultado["fecha"].append(i[8])




    bitcoin_data = resultado




    train = pd.DataFrame.from_dict(resultado)
    train.sort_values(by="fecha")
    values = train.values
    groups = [0, 1, 2, 3, 4, 5, 6]
    i = 1
    


    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(train.columns[group], y=.5, loc='right')
        i += 1
    #pyplot.show()

    import matplotlib.pyplot as plt
    import seaborn as sns

    train=train[['pol_positivo', 'pol_negativo', 'pol_neutral', 'total', 'precio_bitcoin']]

    cor = train.corr()
    print(cor)
    sns.set(style="white")
    f, ax = plt.subplots(figsize=(11, 9))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    ax = sns.heatmap(cor, cmap=cmap, vmax=1, center=0,
                     square=True, linewidths=.5, cbar_kws={"shrink": .7})
    plt.show()




    # prepare data for plot.ly
    data = [go.Scatter(y=resultado["precio_bitcoin"], x=resultado["fecha"])]

    # plot time series data
    #plotly.offline.plot(data)


    plt.plot(train.index, train['pol_positivo'], 'black')
    plt.plot(train.index, train['pol_negativo'], 'yellow')
    plt.plot(train.index, train['pol_neutral'], 'red')
    plt.plot(train.index, train['total'], 'green')

    plt.title('BTC sentiment)')
    plt.xticks(rotation='vertical')
    plt.ylabel("tweets cantidad");
    plt.show();

    plt.plot(train.index, train['precio_bitcoin'], 'g')
    plt.title('Trading Vol BTC(hr)')
    plt.xticks(rotation='vertical')
    plt.ylabel('Vol BTC');
    plt.show();


    values = train.values
    cols = train.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = train[cols]
    train=train[['pol_positivo', 'pol_negativo', 'pol_neutral', 'total', 'precio_bitcoin']]
    df.head()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values)

    n_hours = 1  # adding 3 hours lags creating number of observations
    n_features = 5  # Features in the dataset.
    n_obs = n_hours * n_features

    reframed = series_to_supervised(scaled, n_hours, 1)
    reframed.head()
    print(reframed.head())

    values = reframed.values
    n_train_hours = 350
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    train.shape

    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    test_X, test_y = test[:, :n_obs], test[:, -n_features]

    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    model = Sequential()
    model.add(LSTM(5, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=6,
                        validation_data=(test_X, test_y), verbose=2,
                        shuffle=False, validation_split=0.2)

    plt.show()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_hours * n_features,))
# invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, -4:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = concatenate((test_y, test_X[:, -4:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    mse = (mean_squared_error(inv_y, inv_yhat))
    print('Test MSE: %.3f' % mse)
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    plt.plot(inv_y, label='Real')
    plt.plot(inv_yhat, label='Predicted')
    plt.show()

