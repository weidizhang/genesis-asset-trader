import numpy as np
import pandas as pd

price_field = "Close"
time_multiplier = 1 # Default = 1 (day); use 24 if using hourly data

# Constrain range of data to [-1, 1]
# Done so that this data can be applied to other data sets
# We only care about the resultant shape, not the raw numbers
def condense_data(df_field):
    midpt = (df_field.min() + df_field.max()) / 2
    dist_directional = df_field.max() - midpt

    # Transform data to have a midpoint of 0
    df_field -= midpt
    # Transform data to range of [-1, 1]
    df_field *= 1 / dist_directional

def generate_hlc(df):
    df["HLCAverage"] = (df["High"] + df["Low"] + df["Close"]) / 3

def generate_ema(df, period):
    # period in days
    df["EMA" + str(period)] = pd.Series.ewm(df[price_field], span = period * time_multiplier, adjust = False).mean()

def generate_macd(df):
    # EMA 12 - EMA 26 of price data
    df["MACD"] = pd.Series.ewm(df[price_field], span = 12 * time_multiplier, adjust = False).mean() - pd.Series.ewm(df[price_field], span = 26 * time_multiplier, adjust = False).mean()
    
    # EMA 9 of the MACD
    df["MACDSignal"] = pd.Series.ewm(df["MACD"], span = 9 * time_multiplier, adjust = False).mean()

    df["MACDCrossDifference"] = df["MACD"] - df["MACDSignal"]
    df["MACDCrossDirection"] = np.where(
        np.sign(df["MACDCrossDifference"].shift().fillna(0)) != np.sign(df["MACDCrossDifference"]),
        np.sign(df["MACDCrossDifference"]),
        np.nan
    )

def generate_obv(df):
    for i in range(len(df)):
        volume = df.loc[i, "Volume"]

        if i == 0:
            df.loc[i, "OBV"] = volume
        else:
            prev_obv = df.loc[i - 1, "OBV"]
            prev_close = df.loc[i - 1, price_field]
            close = df.loc[i, price_field]

            df.loc[i, "OBV"] = prev_obv + (volume if close > prev_close else (-volume if close < prev_close else 0))

    # Transform data to have zero as fixed point minimum
    df["OBV"] -= df["OBV"].min()

def generate_rsi(df, period = 14):
    def calculate_rsi(delta):
        gain, loss = delta.copy(), abs(delta.copy())
        gain[delta < 0] = 0
        loss[delta > 0] = 0
        rs = gain.rolling(period * time_multiplier).mean() / loss.rolling(period * time_multiplier).mean()
        return 100 - 100 / (1 + rs)

    delta = df[price_field].diff()
    df["RSI"] = calculate_rsi(delta)
