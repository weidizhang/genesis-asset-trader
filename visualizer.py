from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt

import data_processor

register_matplotlib_converters()
plt.rcParams["figure.figsize"] = (10, 6)

df = data_processor.main()
fig, axs = plt.subplots(3, gridspec_kw = { "height_ratios": [3, 1, 1] })
ax0, ax1, ax2 = axs
x = df["Date"]

# Price, EMA30
ax0.plot(x, df["HLCAverage"])
ax0.plot(x, df["EMA30"])
ax0.legend(["HLC Average", "EMA30"])

# MACD
ax1.plot(x, df["MACD"])
ax1.plot(x, df["MACDSignal"])
ax1.legend(["MACD", "Signal"])

# RSI
ax2.plot(x, df["RSI"])
ax2.legend(["RSI"])

# Hide dates besides last subplot
for i in range(len(axs) - 1):
    axs[i].get_xaxis().set_visible(False)

plt.show()
