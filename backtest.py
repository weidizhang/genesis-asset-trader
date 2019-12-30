from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

from predict import Predict

import backtest_strategy as strategy
import data_processor

def configure():
    register_matplotlib_converters()
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlepad"] = 3

def predict_extremas(data_file_path,
        data_year_range = None, data_hourly = False,
        extrema_n = 20):

    # Use the data processor to get the data frame for actual data
    df = data_processor.read_data(data_file_path, data_year_range, True, data_hourly, extrema_n, False)

    # Use optional keyword for readability
    predict = Predict(
        "models/model.joblib",
        k_neighbors = 5,
        max_conflicts = 2,
        search_distance = 26
    )
    Predict.preprocess_data(df)

    removed_extremas = predict.predict_full(df, Predict.feature_attributes(df))
    print("Extremas removed from originally predicted model via heuristic:", removed_extremas)

    removed_extremas = strategy.alternate_extremas(df)
    print("Extremas removed from originally predicted model via strategy:", removed_extremas)

    strategy.end_with_sell(df)

    return df

def visualize_data(df):
    fig, axs = plt.subplots(2, gridspec_kw = { "height_ratios": [3, 1] })
    ax0, ax1 = axs
    x = df["Date"]

    fig.suptitle("Overall Predictive Backtest")

    # Price
    ax0.title.set_text("Price")

    ax0.plot(x, df["HLCAverage"])

    # Price: Extrema
    minima = df.loc[df["Extrema"] == -1]
    maxima = df.loc[df["Extrema"] == 1]

    ax0.scatter(minima["Date"], minima["HLCAverage"], c = "g")
    ax0.scatter(maxima["Date"], maxima["HLCAverage"], c = "r")

    ax0.legend(["HLC Average", "Model Buy", "Model Sell"])

    visualize_tx_summary(df, ax1, ax0)

    plt.show()

def visualize_tx_summary(df, ax, reference_ax):
    df_summary = strategy.generate_tx_summary(df)
    print(df_summary)

    buy = df_summary.loc[df_summary["Extrema"] == -1]
    sell = df_summary.loc[df_summary["Extrema"] == 1]

    pl_text = str(round(df_summary[::-1].iloc[0]["ProfitLoss"], 2)) + "%"
    ax.title.set_text("Profit/Loss: " + pl_text + " Over Time Span")

    ax.scatter(buy["Date"], buy["ProfitLoss"], c = "g")
    ax.scatter(sell["Date"], sell["ProfitLoss"], c = "r")

    # Replot summary in its entirety to generate a line
    ax.plot(df_summary["Date"], df_summary["ProfitLoss"])

    # Use same range as the price plot by copying a given 
    ax.set_xlim(*reference_ax.get_xlim())

def main():
    data_file_path = "datasets/Coinbase_BTCUSD_1h.csv"
    data_year_range = (2019,)

    df_backtest = predict_extremas(data_file_path, data_year_range, True)
    print(df_backtest)

    configure()
    visualize_data(df_backtest)

if __name__ == "__main__":
    main()
