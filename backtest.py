from predict import Predict
from visualizer import Visualizer

import backtest_strategy as strategy
import data_processor

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

    # This block of code is only for testing the predict_point function in Predict.
    # Uncomment this block of code and remove the call to predict_full following this
    # test block.
    #
    # Observe that its results is the same as using the full prediction.
    # It will run significantly slower than predict_full as it is meant to be used to
    # predict single points in a dataframe only for a faster speed.
    #
    # Begin code block
    #
    # df_features = Predict.feature_attributes(df)
    # results = {}
    # for i in range(len(df)):
    #     results[i] = predict.predict_point(df, df_features, i)
    # for i, e in results.items():
    #     df.loc[i, "Extrema"] = e
    #
    # End code block

    removed_extremas = predict.predict_full(df, Predict.feature_attributes(df))
    print("Extremas removed from originally predicted model via heuristic:", removed_extremas)

    removed_extremas = strategy.alternate_extremas(df)
    print("Extremas removed from originally predicted model via strategy:", removed_extremas)

    strategy.end_with_sell(df)

    return df

def visualize_data(df):
    viz = Visualizer("Overall Predictive Backtest", 2, [3, 1])

    [plot(df, viz) for plot in (plot_price, plot_transaction_summary)]
    viz.show_last_x_axis_only()
    viz.show()

def plot_price(df, viz):
    minima = df.loc[df["Extrema"] == -1]
    maxima = df.loc[df["Extrema"] == 1]

    def price_callback(ax):
        # HLCAverage as price
        ax.plot(df["Date"], df["HLCAverage"])

        # Price Extrema
        ax.scatter(minima["Date"], minima["HLCAverage"], c = "g")
        ax.scatter(maxima["Date"], maxima["HLCAverage"], c = "r")

    viz.fill_next_axis(price_callback, "Price", ["HLC Average", "Model Buy", "Model Sell"])

def plot_transaction_summary(df, viz):
    df_summary = strategy.generate_tx_summary(df)
    print(df_summary)

    buy = df_summary.loc[df_summary["Extrema"] == -1]
    sell = df_summary.loc[df_summary["Extrema"] == 1]

    pl_title = "Profit/Loss: " + str(round(df_summary[::-1].iloc[0]["ProfitLoss"], 2)) + "% Over Time Span"

    def tx_callback(ax):
        # Plot each buy and sell transaction
        ax.scatter(buy["Date"], buy["ProfitLoss"], c = "g")
        ax.scatter(sell["Date"], sell["ProfitLoss"], c = "r")

        # Replot summary in its entirety to generate a line
        ax.plot(df_summary["Date"], df_summary["ProfitLoss"])

        # Use same range as the price plot by copying the limits of the
        # x axis of the price plot (first plot/axis of the figure)
        ax.set_xlim(*viz.get_axes()[0].get_xlim())

    viz.fill_next_axis(tx_callback, pl_title)

def main():
    # Generate predictions / do backtests on the 2019 data, as our model
    # is trained with data from 2017-2018
    data_file_path = "datasets/Coinbase_BTCUSD_1h.csv"
    data_year_range = (2019,)

    df_backtest = predict_extremas(data_file_path, data_year_range, True)
    print(df_backtest)

    visualize_data(df_backtest)

if __name__ == "__main__":
    main()
