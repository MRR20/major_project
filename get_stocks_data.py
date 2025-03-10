import pandas as pd
import numpy as np
import yfinance as yf
import sys

def stocks_data(ticker: str, period: str):
    """
    Args:

    ticker (str) - Stock symbol of company
    period (str) - To return history period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

    """

    # to get all about "ticker"
    stock = yf.Ticker(ticker)

    # get history data of the stock (already in pandas DataFrame)
    data_raw = stock.history(period=period)

    # remove unncessary columns ['Dividends', 'Stock Splits']
    data_raw = data_raw.drop(['Dividends', 'Stock Splits'], axis=1)

    # calculating and defining other stock metrics [EMA_12, EMA_26, MACD, Signal, RSI, CCI, ADX]

    """
    Definations:

    EMA_12 - 12 days Exponential Moving Average
    EMA_26 - 26 days Exponential Moving Average
    MACD - Moving Average Convergence and Divergence (difference of EMA_12 & EMA_26)
    Signal - 9 days EMA of MACD
    RSI - Relative Strength Index (14 days calulation)
    CCI- Used to identify overbought and oversold conditions
    ADX - Measures trend strength based on DMI (Directional Movement Index)

    """

    # calculating TP for CCI
    data_raw['TP'] = (data_raw['High'] + data_raw['Low'] + data_raw['Close']) / 3

    # 12 day & 26 day EMA for MACD
    data_raw['EMA_12'] = data_raw['Close'].ewm(span=12, adjust=False).mean()
    data_raw['EMA_26'] = data_raw['Close'].ewm(span=26, adjust=False).mean()
    data_raw['MACD'] = data_raw['EMA_12'] - data_raw['EMA_26']
    data_raw['Signal'] = data_raw['MACD'].ewm(span=9, adjust=False).mean()

    # RSI calculation
    delta = data_raw['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain/loss
    data_raw['RSI'] = 100 - (100/(1 + rs))

    # CCI calculation
    n = 14
    data_raw['SMA_TP'] = data_raw['TP'].rolling(n).mean()
    data_raw['MAD_TP'] = data_raw['TP'].rolling(n).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    data_raw['CCI'] = (data_raw['TP'] - data_raw['SMA_TP'])/(0.015 * data_raw['MAD_TP'])

    # ADX calculation
    data_raw['+DM'] = data_raw['High'].diff()
    data_raw['-DM'] = data_raw['Low'].diff()
    data_raw['+DM'] = np.where((data_raw['+DM'] > data_raw['-DM']) & (data_raw['+DM'] > 0), data_raw['+DM'], 0)
    data_raw['-DM'] = np.where((data_raw['-DM'] > data_raw['+DM']) & (data_raw['-DM'] > 0), data_raw['-DM'], 0)
    data_raw['TR'] = np.maximum(data_raw['High'] - data_raw['Low'], 
                                np.maximum(abs(data_raw['High'] - data_raw['Close'].shift(1)),
                                           abs(data_raw['Low'] - data_raw['Close'].shift(1))))
    data_raw['ATR'] = data_raw['TR'].rolling(n).mean()
    data_raw['+DI'] = (data_raw['+DM'].rolling(n).mean()/data_raw['ATR']) * 100
    data_raw['-DI'] = (data_raw['-DM'].rolling(n).mean()/data_raw['ATR']) * 100
    data_raw['DX'] = (abs(data_raw['+DI'] - data_raw['-DI'])/(data_raw['+DI'] + data_raw['-DI'])) * 100
    data_raw['ADX'] = data_raw['DX'].rolling(n).mean()


    # getting required data [Open, High, Low, Close, Volume, EMA_12, EMA_26, MACD, Signal, RSI, CCI, ADX]
    data_req = data_raw[['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_12', 'EMA_26', 'MACD', 'Signal', 'RSI', 'CCI', 'ADX']]

    #print(f"Initial Length {len(data_req)}")

    # drop NaN rows as it could be better solution
    # data_req = data_req.dropna()

    # use bfill to save data
    data_req = data_req.bfill()

    #print(f"Present Length {len(data_req)}")

    return data_req


# for testing purpose

if __name__ == "__main__":
    tickers = dow_jones_tickers = [
            "MMM",  # 3M Company
            "AXP",  # American Express
            "AMGN", # Amgen
            "AAPL", # Apple
            "BA",   # Boeing
            "CAT",  # Caterpillar
            "CVX",  # Chevron
            "CSCO", # Cisco Systems
            "KO",   # Coca-Cola
            "DOW",  # Dow Inc.
            "GS",   # Goldman Sachs
            "HD",   # Home Depot
            "HON",  # Honeywell International
            "IBM",  # IBM
            "INTC", # Intel
            "JNJ",  # Johnson & Johnson
            "JPM",  # JPMorgan Chase
            "MCD",  # McDonald's
            "MRK",  # Merck & Co.
            "MSFT", # Microsoft
            "NKE",  # Nike
            "PG",   # Procter & Gamble
            "CRM", #salesforce
            "TRV", #Travelers Companies Inc.
            "UNH", #UnitedHealth Group
            "V",   # Visa
            "VZ",   # Verizon
            "WBA",  # Walgreens Boots Alliance
            "WMT",  # Walmart
            "DIS" #Disney
        ]
    period = "10y"

    for ticker in tickers:
        data = stocks_data(ticker, period)
        data.to_csv(f"./stocks_data/{ticker}_data.csv")

