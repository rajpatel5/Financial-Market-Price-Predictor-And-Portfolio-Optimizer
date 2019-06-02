# Recurrent Neural Network (RNN)
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import price_estimate as price_est
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def update_csv(prices, dates):
    """Updates the CSV file to ony store the Date and Price"""
    with open('prediction.csv', mode='w', newline="") as test_file:
        test_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        test_writer.writerow(['Date', 'Price'])

        for i in range(len(prices)):
            test_writer.writerow([dates[i], prices[i][0]])

    return pd.read_csv('prediction.csv', index_col='Date',
                       parse_dates=True, usecols=['Date', 'Price'], na_values=['nan'])


class Stock:
    def __init__(self, symbol, train, test,future_dates):
        self.stock = symbol
        self.training_data = train
        self.testing_data = test
        self.dates = future_dates

    def predict(self):
        # Part 1 - Data Pre-processing
        # Importing the training set
        dataset_train = self.training_data
        training_set = dataset_train.iloc[:, 1:2].values

        # Feature Scaling
        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(training_set)

        # Creating a data structure with 60 timesteps and 1 output
        x_train = []
        y_train = []
        for i in range(60, len(dataset_train)):
            x_train.append(training_set_scaled[i-60:i, 0])
            y_train.append(training_set_scaled[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshaping
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Part 2 - Building the RNN
        # Initialising the RNN
        regressor = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        # Adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        # Adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.2))
        # Adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.2))
        # Adding the output layer
        regressor.add(Dense(units=1))

        # Compiling the RNN
        regressor.compile(optimizer='adam', loss='mean_squared_error')

        # Fitting the RNN to the Training set
        print("Predicting. . . . .")
        regressor.fit(x_train, y_train, epochs=100, batch_size=32, verbose=0)

        # Part 3 - Making the predictions and visualising the results
        dataset_test = self.testing_data

        # Getting the predicted stock price
        # Estimating future prices up to a month
        dataset = pd.concat((dataset_train, dataset_test), axis=0)
        ohlc_dataset = (dataset['Open'] + dataset['High'] +
                        dataset['Low'] + dataset['Adj Close']) / 4

        price_estimate = price_est.PriceEstimator(ohlc_dataset, self.dates)
        estimate = price_estimate.estimate()

        dataset_test = pd.concat((dataset_test['Adj Close'], estimate.iloc[:]), axis=0)
        dataset_total = pd.concat((dataset_train['Adj Close'], dataset_test.iloc[:]), axis=0)
        inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = sc.transform(inputs)

        x_test = []
        for i in range(60, len(dataset_test) + 60):
            x_test.append(inputs[i-60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predicted_stock_price = regressor.predict(x_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)

        print("Finished Predicting")

        # Creating a dataframe of the predicted prices with corresponding dates
        prediction = update_csv(predicted_stock_price, dataset_test.index)

        # Visualizing the results
        plt.plot(ohlc_dataset.iloc[len(ohlc_dataset) - 126:], color='orange', label="Real Stock Price")
        plt.plot(prediction['Price'], color="blue", label="Predicted Price")
        plt.title(self.stock + ' Stock Price Prediction')
        plt.xticks(rotation='45')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(loc='upper left')
        plt.grid(linestyle='dotted')
        plt.show()
