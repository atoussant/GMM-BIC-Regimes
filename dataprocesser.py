import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


class DataPreprocessor:
    """Handles data preprocessing and dataset creation"""

    def __init__(self, target_column: str, freq: str = "1W", period=None):
        self.target_column = target_column
        self.freq = freq
        self.scaler = None
        self._data = None
        self.length = None
        self.period = period

    @property
    def iloc(self, idx):
        if self._data is None:
            self._data = self.fetch_stock_data()
        return self._data.iloc[idx]

    def fetch_stock_data(self, ticker="NQ=F", shift=3):
        """Fetch stock market data for the target variable."""

        if self.period != "max":
            period_n = datetime.now() - timedelta(days=self.period)
            period_n = period_n.strftime("%Y-%m-%d")  # %H:%M:%S
        print(f"Fetching stock market data for {ticker}...")
        stock_data = yf.download(tickers=ticker, start=period_n, interval=self.freq)
        stock_data["stock_market_return"] = (
            stock_data["Close"].shift(-shift) / stock_data["Close"] - 1
        ) * 100
        stock_data.columns = stock_data.columns.droplevel(level=1)
        stock_data = stock_data.dropna()
        # stock_data = stock_data[["Close"]]#.rename(columns={"Close": "stock_market_return"})
        stock_data.index.name = "date"
        # stock_data = stock_data.resample(self.freq).last()
        tickername = "nasdaq_futures" if ticker == "NQ=F" else ticker
        print(f"\nMax Date for {tickername}: {stock_data.index.max()}")
        if self.length:
            stock_data = stock_data.iloc[-self.length]

        stock_data.to_csv(f"data/{tickername}_{self.freq}_{self.target_column}.csv")
        self._data = stock_data
        return stock_data

    def prepare_data(self, df: pd.DataFrame, is_price: bool = True) -> pd.DataFrame:
        """Prepare data for training"""
        df = df.copy()
        df = df.interpolate("linear", limit_direction="forward")

        if is_price:
            df["log_value"] = np.log(df[self.target_column])
            df["rolling_mean"] = df["log_value"].rolling(window=252).mean()
            df["rolling_std"] = df["log_value"].rolling(window=252).std()
            df[self.target_column] = (df["log_value"] - df["rolling_mean"]) / df[
                "rolling_std"
            ]
            df = df.dropna(axis=0)

        # Convert all columns to float32
        for col in df.columns:
            df[col] = df[col].astype("float32")

        standardization = {
            "mean": df["rolling_mean"].tolist()[-2:],
            "std": df["rolling_std"].tolist()[-2:],
        }
        # print("\nStandardization dictionary:")
        # print(standardization)
        return df[[self.target_column]], standardization
