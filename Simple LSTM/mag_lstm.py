#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:47:32 2021

@author: leo
"""
	

"""
Imports
"""

from pandas import read_csv, datetime, DataFrame, concat, Series, Grouper, period_range, Timestamp
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from math import sqrt
import matplotlib.pyplot as plt
import numpy


"""
Functions
"""

# load dataset example
def parserEx(x):
	return datetime.strptime('190'+x, '%Y-%m')

# load dataset 
def parser(x):
    format = '%Y-%m-%d'
    return datetime.strptime('T'.join(x.split('T', 2)[:1]), format)

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# define network and trains to fit data
def fit_lstm(train, batch_size, nb_epoch, neurons):
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(nb_epoch):
		model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
		model.reset_states()
	return model

# make a prediction
def forecast_lstm(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size=batch_size)
	return yhat[0,0]


"""
Example code with Shampoo sales
"""
"""
# load dataset
seriesEx = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parserEx)

# line plot
seriesEx.plot()
plt.show()

# get raw data
raw_valuesEx = seriesEx.values
diff_valuesEx = difference(raw_valuesEx, 1)

# create supervised set
supervisedEx = timeseries_to_supervised(diff_valuesEx, 1)
supervised_valuesEx = supervisedEx.values

# split data into train and test-sets
trainEx, testEx = supervised_valuesEx[0:-12], supervised_valuesEx[-12:]

# transform scale 
scalerEx, train_scaledEx, test_scaledEx = scale(trainEx, testEx)

# fit the model
lstm_modelEx = fit_lstm(train_scaledEx, 1, 3000, 4)

# forecast the entire training dataset to build up state for forecasting
train_reshapedEx = train_scaledEx[:, 0].reshape(len(train_scaledEx), 1, 1)
lstm_modelEx.predict(train_reshapedEx, batch_size=1)

# walk-forward validation on the test data
predictionsEx = list()
for i in range(len(test_scaledEx)):
	# make one-step forecast
	XEx, yEx = test_scaledEx[i, 0:-1], test_scaledEx[i, -1]
	yhatEx = forecast_lstm(lstm_modelEx, 1, XEx)
	# invert scaling
	yhatEx = invert_scale(scalerEx, XEx, yhatEx)
    # invert differencing
	yhatEx = inverse_difference(raw_valuesEx, yhatEx, len(test_scaledEx)+1-i)
	# store forecast
	predictionsEx.append(yhatEx)
	expectedEx = raw_valuesEx[len(trainEx) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhatEx, expectedEx))
 
# report performance
rmseEx = sqrt(mean_squared_error(raw_valuesEx[-12:], predictionsEx))
print('Test RMSE: %.3f' % rmseEx)
# line plot of observed vs predicted
plt.plot(raw_valuesEx[-12:])
plt.plot(predictionsEx)
plt.show()
"""

"""
Application on magnitude forcasting
"""

series = read_csv('earthquake_data.csv', header=0, usecols=['time','mag'], parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

# line plot
series.plot()
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.show()
plt.close()

# plot number of earthquakes per year
eq_year = series.groupby(series.index.year).count()
plt.bar(eq_year.index.values, eq_year.values)
plt.xlabel('Year')
plt.ylabel('Number of earthquakes')
plt.show()

# remove data after 2011
series = series[:'2010-12-31']

# plot number of earthquakes per year
eq_year = series.groupby(series.index.year).count()
plt.bar(eq_year.index.values, eq_year.values)
plt.xlabel('Year')
plt.ylabel('Number of earthquakes')
plt.show()

# remove data before 1973
series = series['1973-01-01':]

# plot number of earthquakes per year
eq_year = series.groupby(series.index.year).count()
plt.bar(eq_year.index.values, eq_year.values)
plt.xlabel('Year')
plt.ylabel('Number of earthquakes')
plt.show()

eq_month = series.groupby(Grouper(freq="M"))

eq_month_mean = eq_month.mean()
eq_month_mean.fillna(0, inplace=True)

eq_month_count = eq_month.count()

# get raw data
raw_values = eq_month_count.values
diff_values = difference(raw_values, 1)

# create supervised set
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

# split data into train and test-sets
counter = 0
for name, group in eq_month:
    counter +=1
    if '2006-12-31' in str(name):
        break

split_index = len(eq_month) - counter
train, test = supervised_values[0:-split_index], supervised_values[-split_index:]

# transform scale 
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
lstm_model = fit_lstm(train_scaled, 14, 10, 4)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
	# make one-step forecast
	X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
	yhat = forecast_lstm(lstm_model, 14, X)
	# invert scaling
	yhat = invert_scale(scaler, X, yhat)
	# invert differencing
	yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
	# store forecast
	predictions.append(yhat)
	expected = raw_values[len(train) + i + 1]
	print('Month=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[-split_index:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(raw_values[-split_index:])
plt.plot(predictions)
plt.xlabel('Month number')
plt.ylabel('Number of earthquakes')
plt.legend(['Raw values', 'Predictions'], loc = 'upper right')
plt.show()