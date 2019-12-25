import numpy as np
import pandas as pd

def read_data(data_file_path, data_year = None):
    # Data frame
    df = pd.read_csv(
        data_file_path,
        header = 1,
        parse_dates = ["Date"],
        date_parser = lambda x : pd.datetime.strptime(x, "%Y-%m-%d")
    )
    df.drop("Volume USD", axis = 1, inplace = True)
    df.rename(columns = { "Volume BTC": "Volume" }, inplace = True) # Rename the asset volume column to just Volume

    if data_year:
        start_time = pd.datetime(year = data_year, month = 1, day = 1)
        end_time = pd.datetime(year = data_year, month = 12, day = 31)
        df = df[(df.Date >= start_time) & (df.Date <= end_time)]

    # Reverse the data set as it is loaded in date descending order, we want ascending
    df = df.iloc[::-1]
    df.reset_index(drop = True, inplace = True)
    
    generate_ema(df, 30)
    generate_macd(df)

    return df

def generate_ema(df, period):
    # period in days
    df["EMA" + str(period)] = pd.Series.ewm(df["Close"], span = period, adjust = False).mean()

def generate_macd(df):
    # EMA 12 - EMA 26
    df["MACD"] = pd.Series.ewm(df["Close"], span = 12, adjust = False).mean() - pd.Series.ewm(df["Close"], span = 26, adjust = False).mean()
    
    # EMA 9 of the MACD
    df["MACDSignal"] = pd.Series.ewm(df["MACD"], span = 9, adjust = False).mean()

    df["CrossDifference"] = df["MACD"] - df["MACDSignal"]
    df["MACDCrossDirection"] = np.where(
        np.sign(df.CrossDifference.shift(1)) != np.sign(df.CrossDifference),
        np.sign(df.CrossDifference),
        np.nan
    )

    df.drop("CrossDifference", axis = 1, inplace = True)

def main():
    data_file_path = "unprocessed_data/Coinbase_BTCUSD_d.csv"
    training_data_year = 2019

    df = read_data(data_file_path, training_data_year)
    print(df)

if __name__ == "__main__":
    main()
