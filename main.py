import yfinance as yf
import pandas as pd
import datetime as dt
from datetime import datetime, timedelta
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
import concurrent.futures


def is_upward_trend(data, n=25, slope = 0.25):
    """
    this function checks whether or not a stock is in a upward trend in the past n days

    Parameters:
    data: yfinance data
    n: number of days 
    slope: the slope of the stock data in the past n days
    Returns:
    bool: whether or not stock is in a upward trend

    """
    if len(data) < n:
        return False

    # Use the last n days of data for trend analysis
    data = data[-n:]

    # Prepare data for linear regression
    x = np.arange(n).reshape(-1, 1)
    y_highs = data['High'].values
    y_lows = data['Low'].values

    # Perform linear regression on both highs and lows
    reg_highs = LinearRegression().fit(x, y_highs)
    reg_lows = LinearRegression().fit(x, y_lows)

    # Check if both the highs and lows are trending upwards
    if reg_highs.coef_ > slope and reg_lows.coef_ > slope:
        return True

    return False

def is_high_volume(data, i, n=10, co_ef = 1):
    """
    this function checks if a stock has relatively high volume

    Parameters:
    data: yfinance data
    n: number of days 
    co_ef: the coeffcient we use to times the average volume in the past n days
    Returns:
    bool: whether or not stock on that day has high volume

    """
    if i < n:
        return False

    today_volume = data.iloc[i]['Volume']
    avg_volume = data.iloc[i - n:i]['Volume'].mean()

    return today_volume > (avg_volume * co_ef)

def get_ma_month(data):
    return data['Close'].rolling(window=20).mean()

def get_ma_5(data):
    return data["Close"].rolling(window=5).mean()

def get_ma_10(data):
    return data["Close"].rolling(window=10).mean()

def calculate_kdj(stock_data, k_period=9, d_period=3, j_period=3):
    low_min = stock_data['Low'].rolling(window=k_period).min()
    high_max = stock_data['High'].rolling(window=k_period).max()
    rsv = ((stock_data['Close'] - low_min) / (high_max - low_min)) * 100

    # Calculate K using vectorized operations
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    
    # Calculate D using vectorized operations
    d = k.ewm(alpha=1/3, adjust=False).mean()

    # Calculate J using vectorized operations
    j = 3 * k - 2 * d

    return k, d, j

# real function here

def main(params, stock_symbols = sp500_list):

   
    
    n_upward_trend, slope_upward_trend, n_high_volume, co_ef_high_volume, kdj_thresh, sell_thresh, n_maximum_stock = params
    
    """
     'INTC',
        'CMCSA',
        'VZ',
        'T',
        'CSCO',
        'PFE',
        'WMT',
        'XOM',
        'KO',
        'PEP',
        'MRK',
    """



    initial_investment = 100000
    cash = initial_investment
    stocks_owned = {}

    buy_signals = []
    sell_signals = []
    
    start='2019-06-01'
    end='2023-03-23'
    stock_symbols = sp500_list
    nasdaq_data = yf.download('^IXIC', start=start, end=end)
    stock_data_full = yf.download(stock_symbols, start=start, end=end)
    stock_data_full = stock_data_full.swaplevel(axis=1).sort_index(axis=1)

    # takes 48s to run
    def process_stock_symbol(symbol):
        k, d, j = calculate_kdj(stock_data_full[symbol])
        ma_month = get_ma_month(stock_data_full[symbol])
        ma_5 = get_ma_5(stock_data_full[symbol])
        ma_10 = get_ma_10(stock_data_full[symbol])
        return symbol, k, d, j, ma_month, ma_5, ma_10

    # Parallelize the loop
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_stock_symbol, stock_symbols)

    # Assign the results to stock_data_full
    for result in results:
        symbol, k, d, j, ma_month, ma_5, ma_10 = result
        stock_data_full[(symbol, 'K')] = k
        stock_data_full[(symbol, 'D')] = d
        stock_data_full[(symbol, 'J')] = j
        stock_data_full[(symbol, 'MA_month')] = ma_month
        stock_data_full[(symbol, 'MA_5')] = ma_5
        stock_data_full[(symbol, 'MA_10')] = ma_10

#    nasdaq_data = yf.download('^IXIC', start='2022-01-01', end=dt.datetime.now().strftime('%Y-%m-%d'))

    def process_symbol(symbol, i, nasdaq_data):
        stock_data = stock_data_full[symbol]
        row_today = stock_data.iloc[i]
        row_yesterday = stock_data.iloc[i - 1]
        nasdaq_today = nasdaq_data.loc[row_today.name]
        nasdaq_yesterday = nasdaq_data.loc[row_yesterday.name]

        buy_signal = None
        sell_signal = None

        # Check for buy signal
        if row_yesterday['K'] < kdj_thresh and row_yesterday['D'] < kdj_thresh and row_yesterday['K'] < row_yesterday['D'] \
        and row_today['K'] > row_today['D'] and \
        row_today["MA_5"] > row_today["MA_10"] and row_today["MA_10"] > row_today["MA_month"]:

            if is_high_volume(stock_data, i, n = n_high_volume, co_ef = co_ef_high_volume) and \
            is_upward_trend(stock_data.iloc[:i+1], n = n_upward_trend, slope = slope_upward_trend) and \
            is_upward_trend(nasdaq_data.loc[:nasdaq_today.name], n = n_upward_trend, slope = slope_upward_trend):                    
                buy_signal = {'symbol': symbol, 'date': row_today.name, 'price': row_today['Close']}

        # Check for sell signal
        if row_yesterday['Close'] > row_yesterday['MA_month'] * sell_thresh \
        and row_today['Close'] < row_today['MA_month'] * sell_thresh:
            sell_signal = {'symbol': symbol, 'date': row_today.name, 'price': row_today['Close']}

        return buy_signal, sell_signal

    for i in range(1, len(stock_data_full) - 1):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_symbol, stock_symbols, [i]*len(stock_symbols), [nasdaq_data]*len(stock_symbols)))

            for buy_signal, sell_signal in results:
                if buy_signal:
                    symbol = buy_signal['symbol']
                    row_today = stock_data_full[symbol].iloc[i]

                    if len(stocks_owned) < n_maximum_stock:
                        # Buy the stock
                        allocated_cash = cash / (n_maximum_stock - len(stocks_owned)) # Allocate cash evenly to the remaining stock positions
                        num_shares = allocated_cash // row_today['Close']
                        if num_shares > 0:
                            if symbol not in stocks_owned.keys():
                                stocks_owned[symbol] = num_shares
                            else:
                                stocks_owned[symbol] += num_shares

                            cash -= num_shares * row_today['Close']

                            print(stock_data_full[symbol].index[i])
                            print("buy {}stock at:{}".format(symbol,row_today['Close']))
                    buy_signals.append(buy_signal)

                if sell_signal:
                    symbol = sell_signal['symbol']
                    row_today = stock_data_full[symbol].iloc[i]

                    # Sell the stock
                    if symbol in stocks_owned:
                        cash += stocks_owned[symbol] * row_today['Close']
                        start_date = datetime.strptime(start, '%Y-%m-%d')
                        print(stock_data_full[symbol].index[i])
                        print("sell {} stock at:{}".format(symbol, row_today['Close']))
                        del stocks_owned[symbol]
                    sell_signals.append(sell_signal)
                    
    buy_signals_df = pd.DataFrame(buy_signals)
    sell_signals_df = pd.DataFrame(sell_signals)

#    print("Buy signals:")
#    print(buy_signals_df)

#    print("\nSell signals:")
#    print(sell_signals_df)

    # Calculate the total value of stocks and cash
    total_value = cash
    for symbol, shares in stocks_owned.items():

#        ta = yf.download(symbol, start=start, end=end)

        total_value += shares * stock_data_full.loc[stock_data_full.index[-1], (symbol, 'Close')]
#        print(total_value)

    # Calculate the return rate
    return_rate = (total_value - initial_investment) / initial_investment * 100
#    print(f"\nInitial investment: ${initial_investment:.2f}")
#    print(f"Total value: ${total_value:.2f}")
#    print(f"Return rate: {return_rate:.2f}%")

    
    
#    if return_only:
#        return -return_rate
#    else:
        # Print results as before
    print(f"\nInitial investment: ${initial_investment:.2f}")
    print(f"Total value: ${total_value:.2f}")
    print(f"Return rate: {return_rate:.2f}%")
    return buy_signals_df, sell_signals_df
    


params_ = [5,  0, 3,  1.7247925349436912, 59,  0.9357612919167262,3]
