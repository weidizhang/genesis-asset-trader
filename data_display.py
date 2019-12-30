from visualizer import Visualizer
import data_processor

def plot_price(df, viz):
    minima = df.loc[df["Extrema"] == -1]
    maxima = df.loc[df["Extrema"] == 1]

    def price_callback(ax):
        # HLCAverage as price, EMA 30
        ax.plot(df["Date"], df["HLCAverage"])
        ax.plot(df["Date"], df["EMA30"])

        # Price Extrema
        ax.scatter(minima["Date"], minima["HLCAverage"], c = "g")
        ax.scatter(maxima["Date"], maxima["HLCAverage"], c = "r")

    viz.fill_next_axis(price_callback, "Price", ["HLC Average", "EMA30", "Minima", "Maxima"])

def plot_macd(df, viz):
    def macd_callback(ax):
        ax.plot(df["Date"], df["MACD"])
        ax.plot(df["Date"], df["MACDSignal"])

    viz.fill_next_axis(macd_callback, "Moving Average Convergence Divergence", ["MACD", "Signal"])

def plot_obv(df, viz):
    def obv_callback(ax):
        ax.plot(df["Date"], df["OBV"])

    viz.fill_next_axis(obv_callback, "On-Balance Volume", ["OBV"])

def plot_rsi(df, viz):
    def rsi_callback(ax):
        ax.plot(df["Date"], df["RSI"])

    viz.fill_next_axis(rsi_callback, "Relative Strength Indicator", ["RSI"])

def main():
    df = data_processor.main()
    viz = Visualizer("Indicator Movement on Condensed Range Data", 4, [3, 1, 1, 1])

    [plot(df, viz) for plot in (plot_price, plot_macd, plot_obv, plot_rsi)]
    viz.show_last_x_axis_only()
    viz.show()

if __name__ == "__main__":
    main()
