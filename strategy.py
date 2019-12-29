from predict import Predict

# Essentially, alternate buy and sell transactions by alternating minimas
# and maximas; must start with a buy (minima)
#
# Used for backtesting, mimicking behavior of buying or selling on the first
# received minima or maxima signal, and applying the rule that buys and sells
# must alternate, i.e. 3 buys in a row is illegal
#
# Can prohibit selling on a loss (a maxima's price or value must be greater
# than the previous minima's price or value), in the case that a user enforces
# this rule in a real world trading situation
def alternate_extremas(df, sell_at_loss = False):
    remove_extremas = []
    prev_type = 1 # Say the previous type is a maxima so that it must start with a buy (minima)
    prev_value = 0

    for i, row in df.iterrows():
        extrema_type = row["Extrema"]
        value = row["HLCAverage"]
        if extrema_type == 0:
            continue
        elif extrema_type == prev_type or ((not sell_at_loss) and extrema_type == 1 and value < prev_value):
            remove_extremas.append(i)
            continue

        prev_type, prev_value = extrema_type, value

    Predict.delete_extremas(df, remove_extremas)
    return len(remove_extremas)

# Used for backtesting to enforce that the last transaction must be a "sell"
# (maxima) to give a better picture of profitability
#
# Assumes that alternate_extremas was already applied to df
def end_with_sell(df):
    last_extrema = df[df["Extrema"] != 0][::-1].iloc[0]
    if last_extrema["Extrema"] == -1:
        Predict.delete_extremas(df, last_extrema.name)
