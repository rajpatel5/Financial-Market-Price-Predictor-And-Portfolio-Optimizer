import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def print_data(sharpe_portfolio, min_variance_port):
    print("\nMaximum Return Portfolio Allocation (In Terms of Percentage)")
    print("-----------------------------------------------------------------")
    print(sharpe_portfolio.T)
    print("\nMinimum Volatility Portfolio Allocation (In Terms of Percentage)")
    print("-----------------------------------------------------------------")
    print(min_variance_port.T)


def plot_data(df, sharpe_portfolio, min_variance_port):
    plt.style.use('seaborn-dark')
    df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                    cmap='RdYlGn', edgecolors='black', figsize=(7, 5.5), grid=True)
    plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red',
                marker=(5, 1, 0), s=150)
    plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue',
                marker=(5, 1, 0), s=150)
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier')
    plt.show()


class Portfolio:
    def __init__(self, symbols, csv_data):
        self.stocks = symbols
        self.data = csv_data

    def optimize(self):
        self.data.sort_index(inplace=True)

        # Calculate daily and annual returns of the stocks
        returns_daily = self.data.pct_change()
        returns_annual = returns_daily.mean() * 252

        # Get daily and covariance of returns of the stock
        cov_daily = returns_daily.cov()
        cov_annual = cov_daily * 252

        # Empty lists to store returns, volatility and weights of imaginary portfolios
        port_returns = []
        port_volatility = []
        sharpe_ratio = []
        stock_weights = []

        # Set the number of combinations for imaginary portfolios
        num_assets = len(self.stocks)
        num_portfolios = 50000

        # Populate the empty lists with each portfolios returns,risk and weights
        for i in range(num_portfolios):
            if i % 9000 == 0:
                print("Optimizing. . . . .")

            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            returns = np.dot(weights, returns_annual)
            volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
            sharpe = returns / volatility
            sharpe_ratio.append(sharpe)
            port_returns.append(returns * 100)
            port_volatility.append(volatility * 100)
            stock_weights.append(weights)
        print("Finished Optimizing")

        # A dictionary for Returns and Risk values of each portfolio
        portfolio = {'Returns': port_returns, 'Volatility': port_volatility, 'Sharpe Ratio': sharpe_ratio}

        # Extend original dictionary to accomodate each ticker and weight in the portfolio
        for i, symbol in enumerate(self.stocks):
            portfolio[symbol + ' Weight'] = [round(Weight[i] * 100, 2) for Weight in stock_weights]

        # Make a dataframe of the extended dictionary
        df = pd.DataFrame(portfolio)

        # Get better labels for desired arrangement of columns
        column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock + ' Weight' for stock in self.stocks]

        # Re-order dataframe columns
        df = df[column_order]

        # Find min Volatility & max sharpe values in the dataframe
        min_volatility = df['Volatility'].min()
        max_sharpe = df['Sharpe Ratio'].max()

        # Use the min, max values to locate and create the two special portfolios
        sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
        min_variance_port = df.loc[df['Volatility'] == min_volatility]

        # Print the details of the 2 portfolios
        print_data(sharpe_portfolio, min_variance_port)

        # Plot efficient frontier, max sharpe & min volatility values with a scatter plot
        plot_data(df, sharpe_portfolio, min_variance_port)
