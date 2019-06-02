# Market-Price-Predictor-Portfolio-Optimizer

A market price predictor and portfolio optimzer which can be used in the stock and forex markets by traders grow their portfolios. The
portfolio optimzer uses the sharpe ratio to determine the maximum return yeilding asset allocation and also the minimum volatility asset
allocation. Then displays it on a graph. The market price predictor uses machine learning to attempt to quickly and accurately predict the
general trend of the markets for a month's time. This program is able to predict the prices with 60%-65% of the time, therefore theoratically it can help make a trader good money in the markets.

The program is coded in Python using Pycharm. It uses the matplot library to plot the results onto a graph, and the tensorflow library to use machine learning to predict the prices. The pandas library is used to store and retrieve information from a dataframe. The price predictor makes good use of recurrent neural networks and linear regression to predict the future prices. The predicted prices are stored in a csv file which are then later used to plot them on a graph. The program aims to focus on maximum efficiency and speed as the markets themselves are rapidly changing.
