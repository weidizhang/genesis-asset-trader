import pandas as pd

import indicators

def read_data(data_file_path, data_year_range = None):
    # Data frame
    df = pd.read_csv(
        data_file_path,
        header = 1,
        parse_dates = ["Date"],
        date_parser = lambda x : pd.datetime.strptime(x, "%Y-%m-%d")
    )
    df.drop("Volume USD", axis = 1, inplace = True)
    df.rename(columns = { "Volume BTC": "Volume" }, inplace = True) # Rename the asset volume column to just Volume

    if data_year_range:
        start_time = pd.datetime(year = min(data_year_range), month = 1, day = 1)
        end_time = pd.datetime(year = max(data_year_range), month = 12, day = 31)
        df = df[(df.Date >= start_time) & (df.Date <= end_time)]

    # Reverse the data set as it is loaded in date descending order, we want ascending
    df = df.iloc[::-1]
    
    indicators.generate_hlc(df)
    indicators.price_field = "HLCAverage"

    indicators.generate_ema(df, 30)
    indicators.generate_macd(df)
    indicators.generate_rsi(df)

    # Remove first 30 days of entries to improve accuracy of calcuated technical indicators in training data
    df = df[30:]
    df.reset_index(drop = True, inplace = True)

    return df

def main():
    data_file_path = "unprocessed_data/Coinbase_BTCUSD_d.csv"
    training_data_year_range = (2018, 2019)

    df = read_data(data_file_path, training_data_year_range)
    print(df)

    return df

if __name__ == "__main__":
    main()
