import numpy as np
import pandas as pd

data_file_path = "unprocessed_data/Coinbase_BTCUSD_1h.csv"
training_data = "2018" # Only use data from 2018

csv_data = pd.read_csv(
    data_file_path,
    header = 1,
    parse_dates = ["Date"],
    date_parser = lambda x : pd.datetime.strptime(x, "%Y-%m-%d %I-%p")
)
print(csv_data)
