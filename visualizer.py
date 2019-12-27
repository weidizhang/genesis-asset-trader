from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

import data_processor

def configure():
    register_matplotlib_converters()
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlepad"] = 3

def visualize_data():
    df = data_processor.main()
    fig, axs = plt.subplots(4, gridspec_kw = { "height_ratios": [3, 1, 1, 1] })
    ax0, ax1, ax2, ax3 = axs
    x = df["Date"]

    fig.suptitle("Indicator Movement on Condensed Range Data")

    # Price, EMA30
    ax0.title.set_text("Price")

    ax0.plot(x, df["HLCAverage"])
    ax0.plot(x, df["EMA30"])

    # Price: Extrema
    minima = df.loc[df["Extrema"] == -1]
    maxima = df.loc[df["Extrema"] == 1]

    ax0.scatter(minima["Date"], minima["HLCAverage"], c = "g")
    ax0.scatter(maxima["Date"], maxima["HLCAverage"], c = "r")

    ax0.legend(["HLC Average", "EMA30", "Minima", "Maxima"])

    # MACD
    ax1.title.set_text("Moving Average Convergence Divergence")

    ax1.plot(x, df["MACD"])
    ax1.plot(x, df["MACDSignal"])
    ax1.legend(["MACD", "Signal"])

    # OBV
    ax2.title.set_text("On-Balance Volume")

    ax2.plot(x, df["OBV"])
    ax2.legend(["OBV"])

    # RSI
    ax3.title.set_text("Relative Strength Indicator")

    ax3.plot(x, df["RSI"])
    ax3.legend(["RSI"])

    # Hide dates besides last subplot
    for i in range(len(axs) - 1):
        axs[i].get_xaxis().set_visible(False)

    plt.show()

def main():
    configure()
    visualize_data()

if __name__ == "__main__":
    main()
