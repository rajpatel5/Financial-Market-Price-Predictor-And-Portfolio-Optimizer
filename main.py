import pandas_datareader.data as web
from datetime import datetime
import  pandas as pd
from dateutil.relativedelta import relativedelta
import portfolio_optimizer as port_opt
import stock_algorithm as stock_algo
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def parse_date(date):
    """Parses the date to be used in format of mm/dd/yyyy"""
    date = date.strftime("%x")
    return date[:-2] + "20" + date[-2:]


current_date = datetime.now()
year_ago = current_date - relativedelta(years=1)
five_years_ago = current_date - relativedelta(years=5)
month_ago = current_date - relativedelta(months=1)
month_later = current_date + relativedelta(months=1)
dates = pd.date_range(parse_date(current_date), parse_date(month_later))
dates = dates[dates.dayofweek < 5]
dates = dates[1:]  # Gets rid of current date in the list
# List of stocks in portfolio
stocks = []

print("Options")
print("-------------------------------------")
print("1 - Portfolio Optimization")
print("2 - Stock Price Prediction")
print("3 - Exit Program")

while True:
    option = int(input("\nSelect an option (Enter Number): ").strip())

    while option > 3:
        print("\nInvalid input, please try again")
        option = int(input("Select an option: ").strip())

    if option == 1:
        print("\nPlease enter the stocks you would like in your portfolio")
        print("Once finished inserting stocks, just click enter when asked again\n")

        stock = input("Enter Stock Symbol: ").upper().strip()
        stocks.append(stock)
        while stock != "":
            stock = input("Enter Stock Symbol: ").upper().strip()
            stocks.append(stock)
        stocks = stocks[:-1]

        # Download daily price data for each of the stocks in the portfolio
        if len(stocks) != 0:
            print("Retrieving Data. . . . .")
            data = web.DataReader(stocks, data_source='yahoo',
                                  start=parse_date(year_ago), end=parse_date(current_date))['Adj Close']
        print("Finished Retrieving Data\n")
        portfolio = port_opt.Portfolio(stocks, data)
        portfolio.optimize()

    elif option == 2:
        stock = input("Enter Stock Symbol: ").upper().strip()
        print("\nRetrieving Data. . . . .")
        training_data = web.DataReader(stock, data_source='yahoo',
                                       start=parse_date(five_years_ago), end=parse_date(month_ago))
        testing_data = web.DataReader(stock, data_source='yahoo',
                                      start=parse_date(month_ago), end=parse_date(current_date))
        print("Finished Retrieving Data\n")

        stock_prediction = stock_algo.Stock(stock, training_data, testing_data, dates)
        stock_prediction.predict()

    elif option == 3:
        sys.exit()
