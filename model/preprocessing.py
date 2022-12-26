"""Preprocess data before parsing to build dataset"""

import pandas as pd

def create_features(df, max_input_width, input_cols):
    """Creating features and indicators
    
    Arguments:
        max_input_width: (int) maximize size of input sequence
        input_cols: (list) of input columns from config to remove NAs
    """

    # Moving Average & STD
    df["MA"] = df['Close'].rolling(window=max_input_width).mean()
    df["STD"] = df["Close"].rolling(window=max_input_width).std()

    #  Bollinger Bands
    df['Bollinger_Upper'] = df['MA'] + (df['STD'] * 2)
    df['Bollinger_Lower'] = df['MA'] - (df['STD'] * 2)


    # Drop records that contains any NAs for input_cols
    df = df.dropna(axis=0, how='any', subset=input_cols)
    return df

def select_data(df, input_cols, unwanted_cols=None, unwanted_rows=None, start_date=None):
    """Preprocesses data before parsing to WindowGenerator
    Args:
        df: (pandas DataFrame) Data
        input_cols: (list) of columns to select from the df
        # normalize: (bool) whether to normalize input_cols
        unwanted_cols: (list) of columns names to drop from df
        unwanted_rows: (list) of rows to drop from df
        start_date: (str) starting point of the data
    """
    # Starting point for training data
    if start_date is not None:
        df = df[df["Date"] >= start_date]
    # separete Date column
    date = df.pop('Date')
    # remove unwanted columns
    if unwanted_cols is not None:
        df = df.drop(unwanted_cols, axis=1)
    if unwanted_rows is not None:
        df = df.drop(unwanted_rows, axis=0)
    # select columns
    df = df.get(input_cols)
    return df, date


# def create_features(df, input_cols):
#     """Creating features and indicators
    
#     Arguments:
#         input_cols: (list) of input columns from config to remove NAs
#     """

#     # Moving Avergae at different periods
#     df["MA200"] = df['Close'].rolling(window=200).mean()
#     df["MA50"] = df['Close'].rolling(window=50).mean()
#     df["MA14"] = df['Close'].rolling(window=14).mean()
#     df["MA20"] = df['Close'].rolling(window=14).mean()

#     # Moving Average - High, Low, Std
#     df["MA200_low"] = df["Low"].rolling(window=200).min()
#     df["MA14_low"] = df["Low"].rolling(window=14).min()
#     df["MA200_high"] = df["High"].rolling(window=200).max()
#     df["MA14_high"] = df["High"].rolling(window=14).max()
#     df["MA20_std"] = df["Close"].rolling(window=20).std()

#     # Relative Strength Index
#     df['K-ratio'] = 100*((df['Close'] - df['MA14_low']) / (df['MA14_high'] - df['MA14_low']))
#     df['RSI'] = df['K-ratio'].rolling(window=3).mean() 

#     #  Bollinger Bands
#     df['Bollinger_Upper'] = df['MA20'] + (df['MA20_std'] * 2)
#     df['Bollinger_Lower'] = df['MA20'] - (df['MA20_std'] * 2)

#     # Exponential Moving Averages (EMAS) - different periods
#     df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
#     df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()

#     # Moving Average Convergence/Divergence (MACD)
#     df['MACD'] = df['EMA12'] - df['EMA26']

#     # Drop records that contains any NAs for input_cols
#     df = df.dropna(axis=0, how='any', subset=input_cols)
#     return df


