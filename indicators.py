import numpy as np
import pandas as pd

time_multiplier = 1 # Default = 1 day
price_field = "Close"

def use_hourly_mode():
    set_time_multiplier(24)

def use_day_mode():
    set_time_multiplier(1)

def set_time_multiplier(multiplier):
    time_multiplier = multiplier

def set_price_field(field):
    price_field = field

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
        np.sign(df["MACDCrossDifference"].shift(1).fillna(0)) != np.sign(df["MACDCrossDifference"]),
        np.sign(df["MACDCrossDifference"]),
        np.nan
    )

def generate_rsi(df, period = 14):
    def calculate_rsi(delta):
        gain, loss = delta.copy(), abs(delta.copy())
        gain[delta < 0] = 0
        loss[delta > 0] = 0
        rs = gain.rolling(period * time_multiplier).mean() / loss.rolling(period * time_multiplier).mean()
        return 100 - 100 / (1 + rs)

    delta = df[price_field].diff()
    df["RSI"] = calculate_rsi(delta)
