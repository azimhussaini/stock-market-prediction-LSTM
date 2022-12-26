"""Downloading and saving data from Alpha Vantage"""

import os
import argparse
import numpy as np
import pandas as pd
import datetime as dt
import urllib.request, json

from utils import Params



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', metavar='',
                    help="Directory containing the dataset")


if __name__ == '__main__':

    # Load config
    args = parser.parse_args()
    config_path = os.path.join('config.json')
    assert os.path.isfile(config_path), f"No json data config file found {config_path}"
    config = Params(config_path)
    print("Data configuration loaded")
    
    api_key = config.data["api_key"]
    ticker = config.data["ticker"]
    start_date = dt.datetime.strptime(config.data["start_date"], '%Y-%m-%d')
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)
    file_to_save = f"stock_market_data-{ticker}.csv"
    assert os.path.exists(os.path.join(args.data_dir)), "data directory does not exist. Please create one"
    data_path = os.path.join(args.data_dir, file_to_save)

    if not os.path.exists(data_path):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            # extract stock market data
            data = data["Time Series (Daily)"]
            df = pd.DataFrame(columns=['Date', 'Open', 'Low', 'High', 'Close', 'Volume'])
            for k, v, in data.items():
                date = dt.datetime.strptime(k, '%Y-%m-%d')
                data_row = [date.date(), float(v['1. open']), float(v['3. low']), float(v['2. high']), float(v['4. close']), float(v['5. volume'])]
                df.loc[-1, :] = data_row
                df.index = df.index + 1
            # sort values by date
            df = df.sort_values('Date')
            df = df[df["Date"] >= start_date.date()]
            print(f'Data saved to: {data_path}')
            df.to_csv(data_path)

    else:
        print("File already exists. Loading data from csv")
        df = pd.read_csv(data_path)

    train_filename = "train_" + ticker + ".csv"
    test_filename = "test_" + ticker + ".csv"
    train_data_path = os.path.join(args.data_dir, "train_data")
    assert os.path.exists(train_data_path), "'train_data' directory with data directory does not exist. Please create one"
    test_data_path = os.path.join(args.data_dir, "test_data")
    assert os.path.exists(test_data_path), "'test_data' directory with data directory does not exist. Please create one"
    
    train_split = int(df.shape[0] * (1 - config.data["train_test_split"]))

    if not os.path.exists(os.path.join(train_data_path, train_filename)):
        if not os.path.exists(train_data_path):
            os.mkdir(train_data_path)
        train_data = df[:train_split]
        train_data.to_csv(os.path.join(train_data_path, train_filename))
        print(f'Training Data saved to {train_data_path}.')
    else:
        print("Training Data already exists.")

    if not os.path.exists(os.path.join(test_data_path, test_filename)):
        if not os.path.exists(test_data_path):
            os.mkdir(test_data_path)
        test_data = df[train_split:]
        test_data.to_csv(os.path.join(test_data_path, test_filename))
        print(f'Test Data saved to {test_data_path}.')
    else:
        print("Test Data already exists.")



