from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

from predict import Predict

import data_processor
import train

def configure():
    register_matplotlib_converters()
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlepad"] = 3

def predict_extremas(data_file_path,
        data_year_range = None, data_hourly = False,
        extrema_n = 20):

    # Use the data processor to get the data frame for actual data
    df = data_processor.read_data(data_file_path, data_year_range, True, data_hourly, extrema_n)

    predict = Predict("models/model.joblib")
    train.preprocess_data(df)
    removed_extremas = predict.predict_full(df, train.split_data(df)[0])
    print("Extremas removed from originally predicted model:", removed_extremas)

    return df

def visualize_data(df):
    fig, axs = plt.subplots(1, gridspec_kw = { "height_ratios": [3] })
    ax0 = axs
    x = df["Date"]

    fig.suptitle("backtest")

    # Price, EMA30
    ax0.title.set_text("Price")

    ax0.plot(x, df["HLCAverage"])

    # Price: Extrema
    minima = df.loc[df["Extrema"] == -1]
    maxima = df.loc[df["Extrema"] == 1]

    ax0.scatter(minima["Date"], minima["HLCAverage"], c = "g")
    ax0.scatter(maxima["Date"], maxima["HLCAverage"], c = "r")

    ax0.legend(["HLC Average", "Predicted Minima", "Predicted Maxima"])

    plt.show()

def main():
    data_file_path = "datasets/Coinbase_BTCUSD_1h.csv"
    data_year_range = (2019,)

    df_backtest = predict_extremas(data_file_path, data_year_range, True)
    print(df_backtest)

    configure()
    visualize_data(df_backtest)

if __name__ == "__main__":
    main()
