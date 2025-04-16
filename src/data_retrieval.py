import os
import pandas as pd
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web

# data = web.FredReader(["CPIAUCSL", "M2V"], start="2015-01-01").read()

# nq_vol = yf.download("^VXN", multi_level_index=False, start="2018-01-01")
# bond_vol = yf.download("^MOVE", multi_level_index=False, start="2018-01-01")
dim_vol = yf.download("^COR90D", multi_level_index=False, period="5d")

# Define metrics to retrieve
FED_METRICS = {
    "inflation_rate": "CPIAUCSL",  # Consumer Price Index for All Urban Consumers
    "money_velocity_growth": "M2V",  # Velocity of M2 Money Stock
    "unemployment_rate": "UNRATE",  # Unemployment Rate
    "personal_consumption_growth": "PCE",  # Personal Consumption Expenditures
    "yield_curve_spread": "T10Y2Y",  # 10-Year Treasury Constant Maturity Minus 2-Year
    "weekly_economy_index": "WEI",  # Weekly Economic Index
    "fed_rate": "FEDFUNDS",
    "consumber_sentiment": "UMCSENT",
}

CBOE_METRICS = {"nasdaq_volatility": "^VXN", "bond_volatility": "^MOVE"}


def fetch_and_save_metrics(metrics, start_date="2015-01-01", output_dir="data/raw/"):
    """Fetch and save data for a list of metrics."""
    os.makedirs(output_dir, exist_ok=True)
    for metric, series_id in metrics.items():
        print(f"Fetching data for {metric} ({series_id})...")
        # Fetch data as a pandas Series
        data = web.FredReader(series_id, start=start_date).read()
        # Convert to DataFrame and save as CSV
        df = data.reset_index()
        df.columns = ["date", metric]  # Rename columns for consistency
        output_file = os.path.join(output_dir, f"{metric}.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {metric} to {output_file}")


def cboe_metric(metrics, start_date="2015-01-01", output_dir="data/raw/"):
    os.makedirs(output_dir, exist_ok=True)
    for metric, series_id in metrics.items():
        print(f"Fetching data for {metric} ({series_id})...")
        # Fetch data as a pandas Series
        data = yf.download(series_id, start=start_date, multi_level_index=False)
        # Convert to DataFrame and save as CSV
        df = data.reset_index()
        # df.columns = ["date", metric]  # Rename columns for consistency
        output_file = os.path.join(output_dir, f"{metric}.csv")
        df.columns = df.columns.str.lower()
        df.to_csv(output_file, index=False)
        print(f"Saved {metric} to {output_file}")


if __name__ == "__main__":
    # Fetch and save all metrics
    fetch_and_save_metrics(FED_METRICS)
    cboe_metric(CBOE_METRICS)
    print("All data fetched and saved.")
