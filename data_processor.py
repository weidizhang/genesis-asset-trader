from datetime import datetime
import csv

# Supports files from cryptodatadownload.com

data_file_path = "unprocessed_data/Coinbase_BTCUSD_1h.csv"
training_data = "2018" # Only use data from 2018

def calculate_ema(period_days):
    pass

parsed_data = {}

with open(data_file_path) as data_file:
    csv_data = csv.reader(data_file, delimiter = ',')

    line = 0
    for row in csv_data:
        line += 1
        if line < 3:
            continue

        date = row[0]
        if date[:4] != training_data:
            continue

        date = datetime.strptime(date, "%Y-%m-%d %I-%p")

        # ohlc = open-high-low-close
        prices = {
            "o": row[2],
            "h": row[3],
            "l": row[4],
            "c": row[5]
        }
        volume = row[6] # asset volume, not usd volume

        parsed_data[date] = { "prices": prices, "volume": volume }

print(parsed_data)
