import numpy as np
import pandas as pd

import extrema
import indicators

def read_data(data_file_path,
        data_year_range = None, data_condensed = False, data_hourly = False,
        extrema_n = 20):
    # Data frame
    df = pd.read_csv(
        data_file_path,
        header = 1,
        parse_dates = ["Date"],
        date_parser = lambda x : pd.datetime.strptime(x, "%Y-%m-%d" + (" %I-%p" if data_hourly else ""))
    )
    df.rename(columns = { "Volume BTC": "Volume" }, inplace = True) # Rename the asset volume column to just Volume

    if data_year_range:
        start_time = pd.datetime(year = min(data_year_range), month = 1, day = 1)
        end_time = pd.datetime(year = max(data_year_range), month = 12, day = 31)
        df = df[(df.Date >= start_time) & (df.Date <= end_time)]

    # Reverse the data set as it is loaded in date descending order, we want ascending
    df = df.iloc[::-1]
    # Reset index as correct index is depended on by generate_obv
    df.reset_index(drop = True, inplace = True)
    
    indicators.generate_hlc(df)
    indicators.price_field = "HLCAverage"
    if data_hourly:
        indicators.time_multiplier = 24

    # Drop unused columns
    unused = ["Symbol", "Open", "High", "Low", "Close", "Volume USD"]    
    df.drop(unused, axis = 1, inplace = True)

    # Generate indicators
    indicators.generate_ema(df, 30)
    indicators.generate_macd(df)
    indicators.generate_obv(df)
    indicators.generate_rsi(df)

    # Remove first 30 days of entries to improve accuracy of calcuated technical indicators in training data
    df = df[30 * indicators.time_multiplier:]
    df.reset_index(drop = True, inplace = True)

    # Condense ranges of data after data set rows/entries are finalized
    # If we do not condense, then all original indicator values will be represented instead
    if data_condensed:
        condense(df)

    # Generate crossing data after data range is condensed
    indicators.generate_ema_cross(df)
    indicators.generate_macd_cross(df)

    # Generate local price extremas
    minima_indices = extrema.local_minima(df["HLCAverage"], extrema_n)
    maxima_indices = extrema.local_maxima(df["HLCAverage"], extrema_n)

    df["Extrema"] = np.nan
    df.loc[minima_indices, "Extrema"] = -1
    df.loc[maxima_indices, "Extrema"] = 1

    return df

def condense(df):
    # Range [0, 100]
    # OBV
    indicators.condense_data_hundred(df["OBV"])
    # Price related
    indicators.condense_data_hundred(df["EMA30"], df["HLCAverage"])
    indicators.condense_data_hundred(df["HLCAverage"])

    # Range [-1, 1]
    # MACD related
    # MACDSignal must come first; it is transformed based on original MACD range    
    indicators.condense_data(df["MACDSignal"], df["MACD"])
    indicators.condense_data(df["MACD"])

    # RSI is already in range of [0, 100]

def main():
    data_file_path = "unprocessed_data/Coinbase_BTCUSD_1h.csv"
    training_data_year_range = (2019,)
    extrema_n = 20

    df = read_data(data_file_path, training_data_year_range, True, True, extrema_n)
    print(df)

    return df

if __name__ == "__main__":
    main()
