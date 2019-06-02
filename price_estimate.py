import numpy as np
import pandas as pd
import random


def compute_rsi(df, period=14):
    # Get the difference in price from previous step
    delta = df.diff()
    delta = delta[1:]

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0.0] = 0.0
    down[down > 0.0] = 0.0

    # Calculate the EWM Average
    roll_up = up.ewm(com=(period - 1), min_periods=period).mean()
    roll_down = down.abs().ewm(com=(period - 1), min_periods=period).mean()

    # Calculate the RSI
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi.iloc[period:]


def find_trend(ohlc_dataset, rsi):
    rsi_cpy = rsi.copy()
    ohlc_dataset_cpy = ohlc_dataset.copy()
    new_rsi_cpy = rsi_cpy
    new_ohlc_dataset_cpy = ohlc_dataset_cpy
    rsi_cpy = rsi_cpy.iloc[len(rsi) - 20:]  # takes one month of RSI values
    ohlc_dataset_cpy = ohlc_dataset_cpy.iloc[len(ohlc_dataset_cpy) - 20:]  # takes one month of OHLC values
    rsi_cpy_values = list(rsi_cpy.values)
    mid_index = 9
    min_index = rsi_cpy_values.index(min(rsi_cpy_values))
    max_index = rsi_cpy_values.index(max(rsi_cpy_values))
    current_index = -1

    if abs(max_index - mid_index) > abs(min_index - mid_index):
        current_index = min_index
        new_rsi_cpy = rsi_cpy.iloc[current_index:]
        new_ohlc_dataset_cpy = ohlc_dataset_cpy.iloc[current_index:]
        trend = "UP"
    elif abs(max_index - mid_index) < abs(min_index - mid_index):
        current_index = max_index
        new_rsi_cpy = rsi_cpy.iloc[current_index:]
        new_ohlc_dataset_cpy = ohlc_dataset_cpy.iloc[current_index:]
        trend = "DOWN"
    elif abs(max_index - mid_index) == abs(min_index - mid_index):
        if abs(rsi_cpy_values[max_index] - rsi_cpy_values[mid_index]) > \
                abs(rsi_cpy_values[min_index] - rsi_cpy_values[mid_index]):
            current_index = min_index
            new_rsi_cpy = rsi_cpy.iloc[current_index:]
            new_ohlc_dataset_cpy = ohlc_dataset_cpy.iloc[current_index:]
            trend = "UP"
        elif abs(rsi_cpy_values[max_index] - rsi_cpy_values[mid_index]) < \
                abs(rsi_cpy_values[min_index] - rsi_cpy_values[mid_index]):
            current_index = max_index
            new_rsi_cpy = rsi_cpy.iloc[current_index:]
            new_ohlc_dataset_cpy = ohlc_dataset_cpy.iloc[current_index:]
            trend = "DOWN"

    if len(new_rsi_cpy) > 6:
        if new_rsi_cpy.iloc[-1] > 65:
            current_index = max_index
            new_rsi_cpy = rsi_cpy.iloc[current_index:]
            new_ohlc_dataset_cpy = ohlc_dataset_cpy.iloc[current_index:]
            trend = "DOWN"
        elif new_rsi_cpy.iloc[-1] < 35:
            current_index = min_index
            new_rsi_cpy = rsi_cpy.iloc[current_index:]
            new_ohlc_dataset_cpy = ohlc_dataset_cpy.iloc[current_index:]
            trend = "UP"

    x_values = [(x + 1) for x in range(len(new_rsi_cpy))]
    rsi_slope, rsi_intercept = np.polyfit(x_values, new_rsi_cpy, 1)
    price_slope, price_intercept = np.polyfit(x_values, new_ohlc_dataset_cpy, 1)

    if (rsi_slope > 0 and trend == "DOWN") or (rsi_slope < 0 and trend == "UP"):
        rsi_slope *= -1

    return rsi_slope, rsi_intercept, price_slope, price_intercept, current_index, trend


def linear(ohlc_dataset, rsi_slope, rsi_intercept, price_slope, price_intercept, dates,  trend, starting_index):
    rsi_values = []
    price_values = [list(ohlc_dataset.values)[-1]]
    current_index = 0

    for i in range(len(dates)):
        rsi_offset = random.randint(-10, 10)
        price_offset = random.uniform((price_values[-1] / -100), (price_values[-1] / 100))
        rsi_value = (rsi_slope * (starting_index + i)) + rsi_intercept + rsi_offset
        price_value = (price_slope * (starting_index + i)) + price_intercept + price_offset

        rsi_values.append(rsi_value)
        price_values.append(price_value)
        current_index = i  # Used to slice dates list

        if trend == "UP" and rsi_value > 70:
            break
        elif trend == "DOWN" and rsi_value < 30:
            break

    # return pd.DataFrame(data={'Date': dates[:current_index + 1], 'Price': price_values[1:]})
    return pd.Series(data=price_values[1:], index=dates[:current_index + 1])


class PriceEstimator:
    def __init__(self, dataset, date_ranges):
        self.df = dataset
        self.dates = date_ranges

    def estimate(self):
        rsi = compute_rsi(self.df)
        rsi_slope, rsi_intercept, price_slope, price_intercept, starting_index, trend = find_trend(self.df, rsi)
        return linear(self.df, rsi_slope, rsi_intercept, price_slope, price_intercept, self.dates,
                      trend, starting_index)

