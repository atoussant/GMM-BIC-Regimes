import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from scipy.interpolate import interp1d
import json
import os
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def analyze_hurst_patterns(
    h_df,
    price_df=None,
    h_column="H",
    window_size=63,
    z_threshold=1.5,
    reversion_test_size=10,
    volatility_window=15,
    corr_window=15,
    plot_results=True,
):
    """
    Analyze patterns in Hurst exponent values, detect spikes, test for mean reversion,
    and identify potential market stress periods.

    Parameters:
    -----------
    h_df : pandas.DataFrame
        DataFrame with datetime index and Hurst exponent values
    price_df : pandas.DataFrame, optional
        DataFrame with datetime index and price data (for market stress correlation)
    h_column : str
        Column name in h_df containing Hurst exponent values
    window_size : int
        Original window size used to calculate H values
    z_threshold : float
        Z-score threshold for detecting spikes (default 2.0 standard deviations)
    reversion_test_size : int
        Window size for testing mean reversion behavior after spikes
    volatility_window : int
        Window size for calculating price volatility
    corr_window : int
        Window size for calculating correlation metrics
    plot_results : bool
        If True, generate visualization plots

    Returns:
    --------
    dict
        Dictionary containing analysis results:
        - 'mean_reversion_stats': Statistics about H mean reversion properties
        - 'spike_events': DataFrame with detected spikes and their characteristics
        - 'market_stress_periods': DataFrame with identified stress periods
        - 'stress_indicators': Feature DataFrame with stress indication scores
    """
    # Make a copy of the H dataframe to avoid modifying the original
    h_data = h_df.copy()

    if h_column not in h_data.columns:
        raise ValueError(
            f"Column '{h_column}' not found in H dataframe. Available columns: {h_data.columns.tolist()}"
        )

    # Extract the H series
    h_series = h_data[h_column]

    # Basic statistics
    h_mean = h_series.mean()
    h_std = h_series.std()
    h_min = h_series.min()
    h_max = h_series.max()

    print(f"Hurst Exponent Statistics:")
    print(f"Mean: {h_mean:.3f}, Std Dev: {h_std:.3f}")
    print(f"Min: {h_min:.3f}, Max: {h_max:.3f}")
    print(f"Range: {h_max - h_min:.3f}")

    # Calculate rolling statistics
    h_data["h_rolling_mean"] = h_series.rolling(window=window_size).mean()
    h_data["h_rolling_std"] = h_series.rolling(window=window_size).std()
    h_data["h_z_score"] = (h_series - h_data["h_rolling_mean"]) / h_data[
        "h_rolling_std"
    ]

    # Test for stationarity (mean reversion)
    adf_result = adfuller(h_series.dropna())
    is_mean_reverting = (
        adf_result[1] < 0.05
    )  # p-value less than 0.05 suggests stationarity

    print(f"\nMean Reversion Test (ADF):")
    print(f"Test Statistic: {adf_result[0]:.4f}")
    print(f"p-value: {adf_result[1]:.4f}")
    print(
        f"H series is {'mean-reverting' if is_mean_reverting else 'not mean-reverting'}"
    )

    # If mean-reverting, estimate the half-life
    half_life = None
    if is_mean_reverting:
        # Fit AR(1) model to estimate mean reversion speed
        try:
            # Create lagged series
            h_lag = h_series.shift(1)
            h_diff = h_series - h_lag
            h_lag = h_lag.dropna()
            h_diff = h_diff.dropna()

            # Run regression: y(t) - y(t-1) = c + φy(t-1) + ε
            model = AutoReg(h_diff, lags=1, trend="ct")
            results = model.fit()
            phi = -results.params[1]  # The AR coefficient

            # Calculate half-life: ln(2) / φ
            if phi > 0:
                half_life = np.log(2) / phi
                print(f"Estimated half-life of mean reversion: {half_life:.2f} periods")
            else:
                print(
                    "Could not calculate half-life (non-positive mean reversion coefficient)"
                )
        except Exception as e:
            print(f"Error estimating half-life: {str(e)}")

    # Detect spikes in H values
    h_data["spike_up"] = h_data["h_z_score"] > z_threshold
    h_data["spike_down"] = h_data["h_z_score"] < -z_threshold
    h_data["is_spike"] = h_data["spike_up"] | h_data["spike_down"]

    # Analyze spike behavior
    spike_events = []

    for i in range(len(h_data)):
        if h_data["is_spike"].iloc[i]:
            spike_date = h_data.index[i]
            spike_value = h_data[h_column].iloc[i]
            spike_z = h_data["h_z_score"].iloc[i]
            spike_direction = "up" if h_data["spike_up"].iloc[i] else "down"

            # Analyze post-spike behavior (if enough data points remain)
            if i + reversion_test_size < len(h_data):
                # Get values after spike
                post_spike = h_data[h_column].iloc[i + 1 : i + reversion_test_size + 1]
                initial_deviation = spike_value - h_data["h_rolling_mean"].iloc[i]
                final_deviation = (
                    post_spike.iloc[-1]
                    - h_data["h_rolling_mean"].iloc[i + reversion_test_size]
                )

                # Calculate how quickly H reverted to mean
                reversion_ratio = (
                    1.0 - (final_deviation / initial_deviation)
                    if initial_deviation != 0
                    else 0
                )

                # Categorize reversion speed
                if reversion_ratio >= 0.8:
                    reversion_category = "Fast"
                elif reversion_ratio >= 0.5:
                    reversion_category = "Medium"
                elif reversion_ratio >= 0.2:
                    reversion_category = "Slow"
                else:
                    reversion_category = "None"
            else:
                reversion_ratio = None
                reversion_category = "Unknown"

            spike_events.append(
                {
                    "date": spike_date,
                    "value": spike_value,
                    "z_score": spike_z,
                    "direction": spike_direction,
                    "reversion_ratio": reversion_ratio,
                    "reversion_category": reversion_category,
                }
            )

    # Create DataFrame of spike events
    spike_df = pd.DataFrame(spike_events)
    if not spike_df.empty:
        print(f"\nDetected {len(spike_df)} spikes in H values")
        print(f"Up spikes: {len(spike_df[spike_df['direction'] == 'up'])}")
        print(f"Down spikes: {len(spike_df[spike_df['direction'] == 'down'])}")

        # Distribution of reversion categories
        rev_counts = spike_df["reversion_category"].value_counts()
        print("\nMean reversion after spikes:")
        for category, count in rev_counts.items():
            print(f"{category}: {count} ({count/len(spike_df)*100:.1f}%)")
    else:
        print("\nNo spikes detected with current threshold.")

    # Add lag and difference features to analyze momentum and reversals in H
    h_data["h_lag1"] = h_series.shift(1)
    h_data["h_change"] = h_series - h_data["h_lag1"]
    h_data["h_change_pct"] = h_data["h_change"] / h_data["h_lag1"]
    h_data["h_acceleration"] = h_data["h_change"] - h_data["h_change"].shift(1)

    # Calculate regime shifts as significant changes in the rolling mean
    h_data["regime_shift"] = abs(h_data["h_rolling_mean"].diff()) > (
        h_std * 0.5
    )  # Using 0.5 std dev threshold

    # Market stress analysis
    market_stress_periods = []
    stress_indicators = pd.DataFrame(index=h_data.index)

    # Base stress indicators on H patterns alone
    stress_indicators["extreme_h"] = (h_data["h_z_score"].abs() > z_threshold).astype(
        int
    )
    stress_indicators["h_acceleration"] = abs(h_data["h_acceleration"]) / h_std
    stress_indicators["regime_shift"] = h_data["regime_shift"].astype(int)

    # If price data is provided, incorporate market indicators
    if price_df is not None:
        # Ensure same datetime index
        price_aligned = price_df.reindex(h_data.index, method="ffill")

        if "close" in price_aligned.columns:
            # Calculate returns
            price_aligned["returns"] = price_aligned["close"].pct_change()

            # Volatility (rolling standard deviation of returns)
            price_aligned["volatility"] = (
                price_aligned["returns"].rolling(window=volatility_window).std()
            )
            volatility_z = (
                price_aligned["volatility"]
                - price_aligned["volatility"].rolling(window=window_size).mean()
            ) / price_aligned["volatility"].rolling(window=window_size).std()

            # Add to stress indicators
            stress_indicators["volatility_z"] = volatility_z
            stress_indicators["high_volatility"] = (volatility_z > z_threshold).astype(
                int
            )

            # Drawdowns
            price_aligned["rolling_max"] = (
                price_aligned["close"].rolling(window=window_size, min_periods=1).max()
            )
            price_aligned["drawdown"] = (
                price_aligned["close"] / price_aligned["rolling_max"]
            ) - 1.0

            stress_indicators["drawdown"] = abs(price_aligned["drawdown"])
            stress_indicators["severe_drawdown"] = (
                abs(price_aligned["drawdown"]) > 0.1
            ).astype(int)

            # Correlation between H and returns/volatility
            def rolling_correlation(series1, series2, window):
                return series1.rolling(window=window).corr(series2)

            stress_indicators["h_return_corr"] = rolling_correlation(
                h_series, price_aligned["returns"], corr_window
            )
            stress_indicators["h_vol_corr"] = rolling_correlation(
                h_series, price_aligned["volatility"], corr_window
            )

            # Analyze correlation sign changes
            stress_indicators["h_return_corr_sign_change"] = (
                stress_indicators["h_return_corr"]
                * stress_indicators["h_return_corr"].shift(1)
                < 0
            ).astype(int)

            # Calculate sudden H-volatility correlation spikes
            h_vol_corr_z = (
                stress_indicators["h_vol_corr"]
                - stress_indicators["h_vol_corr"].rolling(window=window_size).mean()
            ) / stress_indicators["h_vol_corr"].rolling(
                window=window_size
            ).std().replace(
                0, 1
            )  # Avoid div by zero
            stress_indicators["h_vol_corr_spike"] = (
                abs(h_vol_corr_z) > z_threshold
            ).astype(int)

    # Calculate composite stress score
    stress_columns = [
        col
        for col in stress_indicators.columns
        if col not in ["h_return_corr", "h_vol_corr"]
    ]
    stress_indicators["stress_score"] = stress_indicators[stress_columns].sum(
        axis=1
    ) / len(stress_columns)

    # Identify high stress periods (top quartile of stress scores)
    high_stress_threshold = stress_indicators["stress_score"].quantile(0.75)
    stress_indicators["high_stress"] = (
        stress_indicators["stress_score"] > high_stress_threshold
    ).astype(int)

    # Find continuous periods of high stress
    if not stress_indicators.empty:
        high_stress = stress_indicators["high_stress"]

        # Find start and end of continuous stress periods
        stress_starts = high_stress.diff().eq(1).replace(False, None).dropna().index
        stress_ends = high_stress.diff().eq(-1).replace(False, None).dropna().index

        # Handle case where series begins or ends with stress
        if len(stress_starts) > len(stress_ends):
            stress_ends = stress_ends.append(pd.Index([high_stress.index[-1]]))
        elif len(stress_ends) > len(stress_starts):
            stress_starts = pd.Index([high_stress.index[0]]).append(stress_starts)

        # Create list of stress periods
        for start, end in zip(stress_starts, stress_ends):
            # Only consider periods of sufficient length (at least 3 data points)
            if (end > start) and (
                stress_indicators.loc[start:end, "high_stress"].sum() >= 3
            ):
                period_data = {
                    "start_date": start,
                    "end_date": end,
                    "duration": (end - start).days,
                    "avg_stress_score": stress_indicators.loc[
                        start:end, "stress_score"
                    ].mean(),
                    "max_h": h_data.loc[start:end, h_column].max(),
                    "min_h": h_data.loc[start:end, h_column].min(),
                    "h_range": h_data.loc[start:end, h_column].max()
                    - h_data.loc[start:end, h_column].min(),
                }

                # Add market data if available
                if price_df is not None and "close" in price_aligned.columns:
                    period_prices = price_aligned.loc[start:end]
                    if not period_prices.empty:
                        period_data.update(
                            {
                                "return": (
                                    (
                                        period_prices["close"].iloc[-1]
                                        / period_prices["close"].iloc[0]
                                        - 1
                                    )
                                    if len(period_prices) > 1
                                    else 0
                                ),
                                "max_drawdown": (
                                    period_prices["drawdown"].min()
                                    if "drawdown" in period_prices
                                    else None
                                ),
                                "avg_volatility": (
                                    period_prices["volatility"].mean()
                                    if "volatility" in period_prices
                                    else None
                                ),
                            }
                        )

                market_stress_periods.append(period_data)

    # Create DataFrame of market stress periods
    market_stress_df = pd.DataFrame(market_stress_periods)
    if not market_stress_df.empty:
        print(f"\nIdentified {len(market_stress_df)} market stress periods")
    else:
        print("\nNo market stress periods identified with current thresholds.")

    # Generate plots if requested
    if plot_results:
        plt.figure(figsize=(15, 10))

        # Plot 1: H values time series with spikes highlighted
        plt.subplot(3, 1, 1)
        plt.plot(h_data.index, h_series, label="Hurst Exponent", color="blue")
        plt.plot(
            h_data.index,
            h_data["h_rolling_mean"],
            label="Rolling Mean",
            color="red",
            linestyle="--",
        )

        # Highlight spikes
        spike_mask = h_data["is_spike"]
        plt.scatter(
            h_data[spike_mask].index,
            h_data[spike_mask][h_column],
            color="orange",
            marker="^",
            s=100,
            label="Spikes",
        )

        plt.title("Hurst Exponent Time Series with Detected Spikes")
        plt.legend()
        plt.grid(True)

        # Plot 2: Z-scores with thresholds
        plt.subplot(3, 1, 2)
        plt.plot(h_data.index, h_data["h_z_score"], label="H Z-Score", color="green")
        plt.axhline(
            y=z_threshold,
            color="red",
            linestyle="--",
            label=f"Threshold (+/- {z_threshold})",
        )
        plt.axhline(y=-z_threshold, color="red", linestyle="--")
        plt.title("Hurst Exponent Z-Scores")
        plt.legend()
        plt.grid(True)

        # Plot 3: Stress score
        plt.subplot(3, 1, 3)
        plt.plot(
            stress_indicators.index,
            stress_indicators["stress_score"],
            label="Stress Score",
            color="purple",
        )
        plt.axhline(
            y=high_stress_threshold,
            color="red",
            linestyle="--",
            label="High Stress Threshold",
        )

        # Highlight high stress periods
        for i, row in market_stress_df.iterrows():
            plt.axvspan(row["start_date"], row["end_date"], alpha=0.2, color="red")

        plt.title("Market Stress Score")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"plots/market_stress_score{window_size}.png")

        # Additional plot: H distribution
        plt.figure(figsize=(10, 6))
        plt.hist(h_series.dropna(), bins=20, alpha=0.7)
        plt.axvline(x=0.5, color="red", linestyle="--", label="Random Walk (H=0.5)")
        plt.axvline(
            x=h_mean, color="green", linestyle="-", label=f"Mean H={h_mean:.3f}"
        )
        plt.title("Distribution of Hurst Exponent Values")
        plt.xlabel("Hurst Exponent (H)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/hurst_distribution{window_size}.png")

        # If price data available, plot H vs market indicators
        if price_df is not None and "close" in price_aligned.columns:
            plt.figure(figsize=(15, 12))

            # Plot H vs Price
            plt.subplot(3, 1, 1)
            ax1 = plt.gca()
            ax2 = ax1.twinx()

            ax1.plot(h_data.index, h_series, label="Hurst Exponent", color="blue")
            ax1.set_ylabel("Hurst Exponent", color="blue")

            ax2.plot(
                price_aligned.index,
                price_aligned["close"],
                label="Price",
                color="green",
                alpha=0.7,
            )
            ax2.set_ylabel("Price", color="green")

            plt.title("Hurst Exponent vs Price")
            plt.grid(True)

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            # Plot H vs Volatility
            plt.subplot(3, 1, 2)
            ax1 = plt.gca()
            ax2 = ax1.twinx()

            ax1.plot(h_data.index, h_series, label="Hurst Exponent", color="blue")
            ax1.set_ylabel("Hurst Exponent", color="blue")

            ax2.plot(
                price_aligned.index,
                price_aligned["volatility"],
                label="Volatility",
                color="red",
                alpha=0.7,
            )
            ax2.set_ylabel("Volatility", color="red")

            plt.title("Hurst Exponent vs Volatility")
            plt.grid(True)

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            # Plot H vs Drawdowns
            plt.subplot(3, 1, 3)
            ax1 = plt.gca()
            ax2 = ax1.twinx()

            ax1.plot(h_data.index, h_series, label="Hurst Exponent", color="blue")
            ax1.set_ylabel("Hurst Exponent", color="blue")

            ax2.plot(
                price_aligned.index,
                price_aligned["drawdown"],
                label="Drawdown",
                color="purple",
                alpha=0.7,
            )
            ax2.set_ylabel("Drawdown", color="purple")

            plt.title("Hurst Exponent vs Drawdowns")
            plt.grid(True)

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            plt.tight_layout()
            plt.savefig(f"plots/h_price_vol_draw{window_size}.png")

            # Plot correlation dynamics
            plt.figure(figsize=(15, 6))

            plt.subplot(2, 1, 1)
            plt.plot(
                stress_indicators.index,
                stress_indicators["h_return_corr"],
                label="H-Returns Correlation",
                color="blue",
            )
            plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            plt.title("Correlation Between Hurst Exponent and Returns")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(
                stress_indicators.index,
                stress_indicators["h_vol_corr"],
                label="H-Volatility Correlation",
                color="red",
            )
            plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            plt.title("Correlation Between Hurst Exponent and Volatility")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"plots/corr_h_returns_vol_{window_size}.png")

            # Create scatter plots to analyze relationships
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.scatter(h_series, price_aligned["returns"], alpha=0.5)
            plt.title("H vs Returns")
            plt.xlabel("Hurst Exponent")
            plt.ylabel("Returns")
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.scatter(h_series, price_aligned["volatility"], alpha=0.5)
            plt.title("H vs Volatility")
            plt.xlabel("Hurst Exponent")
            plt.ylabel("Volatility")
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.scatter(h_series, price_aligned["drawdown"], alpha=0.5)
            plt.title("H vs Drawdown")
            plt.xlabel("Hurst Exponent")
            plt.ylabel("Drawdown")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"plots/scatter_h_returns_vol_draw_{window_size}.png")

    # Compile the results
    mean_reversion_stats = {
        "is_mean_reverting": is_mean_reverting,
        "adf_statistic": adf_result[0],
        "adf_pvalue": adf_result[1],
        "half_life": half_life,
        "mean": h_mean,
        "std_dev": h_std,
        "min": h_min,
        "max": h_max,
    }

    # Return comprehensive analysis results
    return {
        "mean_reversion_stats": mean_reversion_stats,
        "spike_events": spike_df if not spike_df.empty else None,
        "market_stress_periods": (
            market_stress_df if not market_stress_df.empty else None
        ),
        "stress_indicators": stress_indicators,
        "h_data": h_data,
    }


def regime_detection_from_h(
    h_df,
    price_df=None,
    h_column="H",
    n_regimes=5,
    window_size=30,
    min_regime_duration=5,
    plot_results=True,
):
    """
    Detect market regimes based on Hurst exponent values using clustering. Analyze patterns in Hurst exponent values, detect spikes, test for mean reversion,
    and identify potential market stress periods.

    Parameters:
    -----------
    h_df : pandas.DataFrame
        DataFrame with datetime index and Hurst exponent values
    price_df : pandas.DataFrame, optional
        DataFrame with datetime index and price data (for regime characterization)
    h_column : str
        Column name in h_df containing Hurst exponent values
    n_regimes : int
        Number of regimes to identify
    min_regime_duration : int
        Minimum number of consecutive periods required to constitute a regime
    plot_results : bool
        If True, generate visualization plots

    Returns:
    --------
    dict
        Dictionary containing regime analysis results:
        - 'regime_assignments': Series of regime labels for each time period
        - 'regime_stats': DataFrame with statistics for each regime
        - 'transitions': DataFrame showing regime transition probabilities
    """
    from sklearn.cluster import KMeans

    # Make a copy of the H dataframe to avoid modifying the original
    h_data = h_df.copy()

    if h_column not in h_data.columns:
        raise ValueError(f"Column '{h_column}' not found in H dataframe")

    # Extract the H series
    h_series = h_data[h_column].dropna()

    # Prepare features for clustering
    # We'll use H and lagged H to capture both level and trend
    features = pd.DataFrame(index=h_series.index)
    features["H"] = h_series
    features["H_lag1"] = h_series.shift(1)
    features["H_change"] = features["H"] - features["H_lag1"]

    # Drop rows with NaN (from the lag operation)
    features = features.dropna()
    # new_df = features.drop('regime_name', axis=1)
    # new_df.set_index('Date', inplace=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(features.fillna(0))

    gmm = GaussianMixture(
        n_components=n_regimes, covariance_type="full", random_state=42, n_init=10
    )
    gmm.fit(X)
    with open("models/gmm_regimes.pkl", "wb") as f:
        pickle.dump(gmm, f)

    features["regime"] = gmm.predict(X)

    # gmm = GaussianMixture(
    #     n_components=n_regimes,
    #     covariance_type='full',
    #     random_state=42,
    #     n_init=10
    # )

    # regime_indices = gmm.fit_predict(X)
    # Apply K-means clustering
    # kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    # features['regime'] = kmeans.fit_predict(features[['H', 'H_change']])

    # Order regimes by mean H value for interpretability
    regime_stats = (
        features.groupby("regime")["H"].agg(["mean", "std", "count"]).reset_index()
    )
    regime_stats = regime_stats.sort_values("mean")

    # Create mapping from original cluster labels to ordered labels
    regime_mapping = {
        old_label: new_label
        for new_label, old_label in enumerate(regime_stats["regime"].values)
    }

    # Apply mapping to get ordered regime labels
    features["regime"] = features["regime"].map(regime_mapping)

    # Smooth out short regime changes (eliminate noise)
    def smooth_regimes(regime_series, min_duration=min_regime_duration):
        smoothed = regime_series.copy()
        current_regime = smoothed.iloc[0]
        current_count = 1

        for i in range(1, len(smoothed)):
            if smoothed.iloc[i] == current_regime:
                current_count += 1
            else:
                # Check if we need to revert a short regime
                if current_count < min_duration:
                    # Set all previous values back to the new regime
                    start_idx = i - current_count
                    smoothed.iloc[start_idx:i] = smoothed.iloc[i]

                # Reset for new regime
                current_regime = smoothed.iloc[i]
                current_count = 1

        return smoothed

    # Apply smoothing
    features["smoothed_regime"] = smooth_regimes(features["regime"])

    # Add regime labels back to original H dataframe
    regime_assignments = pd.Series(index=h_data.index, dtype="float64")
    regime_assignments.loc[features.index] = features["smoothed_regime"]
    h_data["regime"] = regime_assignments

    # Analyze regime characteristics
    regime_stats = (
        features.groupby("smoothed_regime")
        .agg({"H": ["mean", "std", "min", "max", "count"], "H_change": ["mean", "std"]})
        .reset_index()
    )

    # Rename columns for clarity
    regime_stats.columns = [
        "regime",
        "h_mean",
        "h_std",
        "h_min",
        "h_max",
        "count",
        "h_change_mean",
        "h_change_std",
    ]

    # Calculate regime durations
    regime_durations = []
    regime_shifts = features["smoothed_regime"].diff().fillna(0) != 0
    shift_indices = regime_shifts[regime_shifts].index

    if len(shift_indices) > 0:
        # Add start and end of the series
        all_shift_points = (
            [features.index[0]] + list(shift_indices) + [features.index[-1]]
        )

        for i in range(len(all_shift_points) - 1):
            start_date = all_shift_points[i]
            end_date = all_shift_points[i + 1]

            # For shifts, the regime is the one after the shift
            if i == 0:
                regime = features.loc[start_date, "smoothed_regime"]
            else:
                regime = features.loc[features.index > start_date].iloc[0][
                    "smoothed_regime"
                ]

            duration = (end_date - start_date).days

            regime_durations.append(
                {
                    "regime": regime,
                    "start_date": start_date,
                    "end_date": end_date,
                    "duration_days": duration,
                }
            )

    duration_df = pd.DataFrame(regime_durations)

    # Calculate regime transition probabilities
    transitions = pd.crosstab(
        features["smoothed_regime"].shift(),
        features["smoothed_regime"],
        normalize="index",
    )

    # Analyze market behavior during each regime if price data is available
    regime_market_stats = None
    if price_df is not None:
        # Align price data with regime assignments
        aligned_price = price_df.reindex(features.index, method="ffill")
        if "close" in aligned_price.columns:
            # Calculate returns
            aligned_price["returns"] = aligned_price["close"].pct_change().fillna(0)
            aligned_price["regime"] = features["smoothed_regime"]

            # Group by regime and calculate statistics
            regime_market_stats = aligned_price.groupby("regime").agg(
                {
                    "returns": [
                        "mean",
                        "std",
                        lambda x: (x > 0).mean(),
                        lambda x: x.autocorr(1),
                    ],
                    "close": [
                        lambda x: (x.iloc[-1] / x.iloc[0]) - 1 if len(x) > 1 else 0
                    ],
                }
            )

            # Rename columns for clarity
            regime_market_stats.columns = [
                "avg_daily_return",
                "volatility",
                "win_rate",
                "return_autocorr",
                "total_return",
            ]

            # Format as percentages for better readability
            for col in ["avg_daily_return", "volatility", "win_rate", "total_return"]:
                regime_market_stats[col] = regime_market_stats[col] * 100

            # Calculate annualized metrics
            regime_market_stats["annualized_return"] = (
                regime_market_stats["avg_daily_return"] * 252
            )
            regime_market_stats["annualized_volatility"] = regime_market_stats[
                "volatility"
            ] * np.sqrt(252)
            regime_market_stats["sharpe_ratio"] = (
                regime_market_stats["annualized_return"]
                / regime_market_stats["annualized_volatility"]
            )

            # Calculate regime duration statistics
            regime_durations = (
                features.groupby(
                    (
                        features["smoothed_regime"].shift()
                        != features["smoothed_regime"]
                    ).cumsum()
                )
                .agg(
                    {
                        "smoothed_regime": "first",
                    }
                )
                .assign(
                    count=lambda x: x.index.size,
                    first_date=lambda x: x.index.min(),
                    last_date=lambda x: x.index.max(),
                )
            )

            regime_durations.columns = ["regime", "duration", "start_date", "end_date"]
            avg_duration = regime_durations.groupby("regime")["duration"].mean()

            # Add duration to market stats
            for regime in avg_duration.index:
                if regime in regime_market_stats.index:
                    regime_market_stats.loc[regime, "avg_duration"] = avg_duration[
                        regime
                    ]

    # Generate plots if requested
    if plot_results:
        # Plot regimes over time with H values
        plt.figure(figsize=(15, 8))

        # Plot H series
        plt.subplot(2, 1, 1)
        plt.plot(h_series.index, h_series, color="blue", label="Hurst Exponent")

        # Shade the background according to regime
        if not duration_df.empty:
            for _, row in duration_df.iterrows():
                plt.axvspan(
                    row["start_date"],
                    row["end_date"],
                    alpha=0.2,
                    color=plt.cm.tab10(row["regime"]),
                    label=(
                        f"Regime {row['regime']}"
                        if row["regime"] not in plt.gca().get_legend_handles_labels()[1]
                        else ""
                    ),
                )

        plt.title("Hurst Exponent with Market Regimes")
        plt.ylabel("Hurst Exponent (H)")
        plt.legend()
        plt.grid(True)

        # Plot regime assignment as a discrete series
        plt.subplot(2, 1, 2)
        plt.plot(
            features.index,
            features["smoothed_regime"],
            drawstyle="steps-post",
            color="red",
            linewidth=2,
        )
        plt.yticks(range(n_regimes))
        plt.title("Market Regime Classification")
        plt.ylabel("Regime")
        plt.xlabel("Date")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"plots/regime_class_{window_size}.png")

        # Plot regime characteristics
        plt.figure(figsize=(12, 6))

        for regime in range(n_regimes):
            regime_data = features[features["smoothed_regime"] == regime]["H"]

            # Use kernel density estimation for smoother distribution
            if len(regime_data) > 1:
                # Create KDE distribution
                x = np.linspace(h_series.min(), h_series.max(), 1000)
                kde = stats.gaussian_kde(regime_data)
                y = kde(x)

                # Scale the density to make it comparable across regimes
                y = (
                    y
                    / y.max()
                    * regime_stats[regime_stats["regime"] == regime]["count"].values[0]
                )

                plt.plot(
                    x,
                    y,
                    label=f"Regime {regime}",
                    color=plt.cm.tab10(regime),
                    linewidth=2,
                )

                # Add vertical line at regime mean
                regime_mean = regime_data.mean()
                plt.axvline(
                    x=regime_mean, color=plt.cm.tab10(regime), linestyle="--", alpha=0.7
                )
                plt.text(
                    regime_mean,
                    0,
                    f"{regime_mean:.3f}",
                    color=plt.cm.tab10(regime),
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        plt.title("Hurst Exponent Distribution by Regime")
        plt.xlabel("Hurst Exponent (H)")
        plt.ylabel("Density-weighted Frequency")
        plt.axvline(x=0.5, color="black", linestyle="-", label="Random Walk (H=0.5)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/h_distribution_by_regime_{window_size}.png")

        # If price data is available, plot price with regime background
        if price_df is not None and "close" in price_df.columns:
            aligned_price = price_df.reindex(h_data.index, method="ffill")

            plt.figure(figsize=(15, 8))

            # Set up price plot with regime background
            plt.plot(
                aligned_price.index, aligned_price["close"], color="blue", linewidth=1.5
            )

            # Shade the background according to regime
            if not duration_df.empty:
                for _, row in duration_df.iterrows():
                    plt.axvspan(
                        row["start_date"],
                        row["end_date"],
                        alpha=0.2,
                        color=plt.cm.tab10(row["regime"]),
                        label=(
                            f"Regime {row['regime']}"
                            if row["regime"]
                            not in plt.gca().get_legend_handles_labels()[1]
                            else ""
                        ),
                    )

            plt.title("Asset Price with Market Regimes")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"plots/prices_regimes_{window_size}.png")

            # Plot regime performance metrics
            if regime_market_stats is not None:
                metrics = ["avg_daily_return", "volatility", "win_rate", "sharpe_ratio"]

                plt.figure(figsize=(14, 10))

                for i, metric in enumerate(metrics):
                    plt.subplot(2, 2, i + 1)

                    # Create bar plot
                    bars = plt.bar(
                        regime_market_stats.index,
                        regime_market_stats[metric],
                        color=[plt.cm.tab10(i) for i in regime_market_stats.index],
                    )

                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height,
                            f"{height:.2f}",
                            ha="center",
                            va="bottom",
                        )

                    plt.title(f'Regime {metric.replace("_", " ").title()}')
                    plt.xlabel("Regime")
                    plt.xticks(regime_market_stats.index)
                    plt.grid(True, axis="y")

                plt.tight_layout()
                plt.savefig(f"plots/regime_stats_{window_size}.png")

                # Plot transition probabilities as a heatmap
                plt.figure(figsize=(8, 6))
                plt.imshow(transitions, cmap="YlOrRd", aspect="equal")

                # Add text annotations
                for i in range(transitions.shape[0]):
                    for j in range(transitions.shape[1]):
                        if not np.isnan(transitions.iloc[i, j]):
                            plt.text(
                                j,
                                i,
                                f"{transitions.iloc[i, j]:.2f}",
                                ha="center",
                                va="center",
                                color=(
                                    "black" if transitions.iloc[i, j] < 0.7 else "white"
                                ),
                            )

                plt.colorbar(label="Transition Probability")
                plt.title("Regime Transition Probabilities")
                plt.xlabel("To Regime")
                plt.ylabel("From Regime")
                plt.xticks(range(n_regimes))
                plt.yticks(range(n_regimes))
                plt.tight_layout()
                plt.savefig(f"plots/regime_trans_{window_size}.png")

    # Compile the results
    results = {
        "regime_assignments": h_data["regime"],
        "regime_stats": regime_stats,
        "transitions": transitions,
        "duration_data": duration_df if not duration_df.empty else None,
        "market_stats": regime_market_stats,
    }

    return results


def detect_market_resilience(
    h_df,
    price_df,
    h_column="Hurst",
    window_size=30,
    threshold_percentile=90,
    min_period_length=5,
    plot_results=True,
):
    """
    Analyze market resilience using the Hurst exponent and price data. Detect market regimes based on Hurst exponent values using clustering.
    Analyze patterns in Hurst exponent values, detect spikes, test for mean reversion, and identify potential market stress periods.

    Parameters:
    -----------
    h_df : pandas.DataFrame
        DataFrame with datetime index and Hurst exponent values
    price_df : pandas.DataFrame
        DataFrame with datetime index and price data
    h_column : str
        Column name in h_df containing Hurst exponent values
    window_size : int
        Rolling window size for calculating resilience metrics
    threshold_percentile : float
        Percentile threshold for identifying high/low resilience periods
    min_period_length : int
        Minimum number of consecutive days to consider a resilience period
    plot_results : bool
        If True, generate visualization plots

    Returns:
    --------
    dict
        Dictionary containing resilience analysis results
    """
    # Make copies to avoid modifying originals
    h_data = h_df.copy()
    price_data = price_df.copy()

    # Align data
    aligned_price = price_data.reindex(h_data.index, method="ffill")

    # Create a resilience dataframe
    resilience = pd.DataFrame(index=h_data.index)
    resilience["H"] = h_data[h_column]

    # Calculate returns and volatility
    if "close" in aligned_price.columns:
        aligned_price["returns"] = aligned_price["close"].pct_change()
        aligned_price["volatility"] = (
            aligned_price["returns"].rolling(window=window_size).std()
        )

        # Calculate drawdowns
        aligned_price["rolling_max"] = (
            aligned_price["close"].rolling(window=window_size, min_periods=1).max()
        )
        aligned_price["drawdown"] = (
            aligned_price["close"] / aligned_price["rolling_max"]
        ) - 1.0

        # Add to resilience dataframe
        resilience["returns"] = aligned_price["returns"]
        resilience["volatility"] = aligned_price["volatility"]
        resilience["drawdown"] = aligned_price["drawdown"]

    # Calculate resilience metrics

    # 1. Anti-fragility metric: How well price recovers after drawdowns
    resilience["recovery_strength"] = (
        resilience["returns"]
        .rolling(window=window_size)
        .apply(
            lambda x: np.sum(x[x > 0])
            / (abs(np.sum(x[x < 0])) + 1e-10)  # Ratio of up moves to down moves
        )
    )

    # 2. Consistency metric: Ratio of H to volatility
    resilience["h_vol_ratio"] = resilience["H"] / (resilience["volatility"] + 1e-10)

    # 3. Trend strength: Rolling correlation between H and price
    def rolling_correlation(series1, series2, window):
        return series1.rolling(window=window).corr(series2)

    resilience["h_price_corr"] = rolling_correlation(
        resilience["H"], aligned_price["close"], window_size
    )

    # 4. Composite resilience score
    # Normalize components to 0-1 range
    for col in ["recovery_strength", "h_vol_ratio"]:
        if col in resilience.columns:
            min_val = resilience[col].quantile(0.05)  # 5th percentile to avoid outliers
            max_val = resilience[col].quantile(
                0.95
            )  # 95th percentile to avoid outliers
            resilience[f"{col}_norm"] = (resilience[col] - min_val) / (
                max_val - min_val
            )
            resilience[f"{col}_norm"] = resilience[f"{col}_norm"].clip(
                0, 1
            )  # Clip to 0-1 range

    # Calculate composite score
    norm_columns = [col for col in resilience.columns if col.endswith("_norm")]
    if norm_columns:
        resilience["resilience_score"] = resilience[norm_columns].mean(axis=1)

        # Identify high and low resilience periods
        high_threshold = resilience["resilience_score"].quantile(
            threshold_percentile / 100
        )
        low_threshold = resilience["resilience_score"].quantile(
            1 - threshold_percentile / 100
        )

        resilience["high_resilience"] = resilience["resilience_score"] > high_threshold
        resilience["low_resilience"] = resilience["resilience_score"] < low_threshold

        # Find continuous periods
        def identify_continuous_periods(series, min_length=min_period_length):
            periods = []
            in_period = False
            start_idx = None

            for i, value in enumerate(series):
                if value and not in_period:
                    # Start of a new period
                    in_period = True
                    start_idx = i
                elif not value and in_period:
                    # End of a period
                    if i - start_idx >= min_length:
                        periods.append((series.index[start_idx], series.index[i - 1]))
                    in_period = False

            # Check if we ended in a period
            if in_period and len(series) - start_idx >= min_length:
                periods.append((series.index[start_idx], series.index[-1]))

            return periods

        high_resilience_periods = identify_continuous_periods(
            resilience["high_resilience"]
        )
        low_resilience_periods = identify_continuous_periods(
            resilience["low_resilience"]
        )

        # Plot results if requested
        if plot_results:
            plt.figure(figsize=(15, 10))

            # Plot price
            plt.subplot(3, 1, 1)
            plt.plot(
                aligned_price.index, aligned_price["close"], label="Price", color="blue"
            )

            # Highlight high resilience periods
            for start, end in high_resilience_periods:
                plt.axvspan(
                    start,
                    end,
                    alpha=0.2,
                    color="green",
                    label=(
                        "High Resilience"
                        if start == high_resilience_periods[0][0]
                        else None
                    ),
                )

            # Highlight low resilience periods
            for start, end in low_resilience_periods:
                plt.axvspan(
                    start,
                    end,
                    alpha=0.2,
                    color="red",
                    label=(
                        "Low Resilience"
                        if start == low_resilience_periods[0][0]
                        else None
                    ),
                )

            plt.title("Price with Resilience Periods")
            plt.legend()
            plt.grid(True)

            # Plot H
            plt.subplot(3, 1, 2)
            plt.plot(
                resilience.index,
                resilience["H"],
                label="Hurst Exponent",
                color="purple",
            )
            plt.axhline(
                y=0.5, color="black", linestyle="--", label="Random Walk (H=0.5)"
            )
            plt.title("Hurst Exponent")
            plt.legend()
            plt.grid(True)

            # Plot resilience score
            plt.subplot(3, 1, 3)
            plt.plot(
                resilience.index,
                resilience["resilience_score"],
                label="Resilience Score",
                color="orange",
            )
            plt.axhline(
                y=high_threshold, color="green", linestyle="--", label="High Threshold"
            )
            plt.axhline(
                y=low_threshold, color="red", linestyle="--", label="Low Threshold"
            )
            plt.title("Market Resilience Score")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"plots/price_resil_{window_size}.png")

            # Plot component metrics
            plt.figure(figsize=(15, 10))

            metrics = ["recovery_strength", "h_vol_ratio", "h_price_corr"]
            for i, metric in enumerate(metrics):
                if metric in resilience.columns:
                    plt.subplot(3, 1, i + 1)
                    plt.plot(
                        resilience.index,
                        resilience[metric],
                        label=metric.replace("_", " ").title(),
                    )
                    plt.title(f'{metric.replace("_", " ").title()}')
                    plt.legend()
                    plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"plots/recov_vol_ratio_corr_{window_size}.png")

            # Compare performance in different resilience regimes
            if "returns" in resilience.columns:
                # Create period labels
                period_labels = pd.Series(index=resilience.index, data="Normal")

                for start, end in high_resilience_periods:
                    period_labels.loc[start:end] = "High"

                for start, end in low_resilience_periods:
                    period_labels.loc[start:end] = "Low"

                resilience["period_type"] = period_labels

                # Calculate performance metrics by period type
                performance = resilience.groupby("period_type")["returns"].agg(
                    [
                        ("Avg Return (%)", lambda x: x.mean() * 100),
                        ("Volatility (%)", lambda x: x.std() * 100),
                        ("Positive Days (%)", lambda x: (x > 0).mean() * 100),
                        (
                            "Sharpe Ratio",
                            lambda x: (
                                (x.mean() / x.std()) * np.sqrt(252)
                                if x.std() != 0
                                else 0
                            ),
                        ),
                        ("Cumulative Return (%)", lambda x: ((1 + x).prod() - 1) * 100),
                        (
                            "Max Drawdown (%)",
                            lambda x: (
                                (1 + x).cumprod() / (1 + x).cumprod().cummax() - 1
                            ).min()
                            * 100,
                        ),
                    ]
                )

                # Plot performance comparison
                plt.figure(figsize=(14, 8))

                metrics = [
                    "Avg Return (%)",
                    "Volatility (%)",
                    "Positive Days (%)",
                    "Sharpe Ratio",
                ]
                for i, metric in enumerate(metrics):
                    plt.subplot(2, 2, i + 1)

                    colors = {"High": "green", "Normal": "blue", "Low": "red"}
                    bars = plt.bar(
                        performance.index,
                        performance[metric],
                        color=[colors[idx] for idx in performance.index],
                    )

                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height,
                            f"{height:.2f}",
                            ha="center",
                            va="bottom",
                        )

                    plt.title(metric)
                    plt.grid(True, axis="y")

                plt.tight_layout()
                plt.savefig(f"plots/returns_stats_{window_size}.png")

        # Compile results
        resilience_periods = {
            "high_resilience_periods": high_resilience_periods,
            "low_resilience_periods": low_resilience_periods,
        }

        if "returns" in resilience.columns and "period_type" in resilience.columns:
            performance_metrics = resilience.groupby("period_type")["returns"].agg(
                [
                    ("mean_return", lambda x: x.mean()),
                    ("volatility", lambda x: x.std()),
                    ("win_rate", lambda x: (x > 0).mean()),
                    (
                        "sharpe",
                        lambda x: (
                            (x.mean() / x.std()) * np.sqrt(252) if x.std() != 0 else 0
                        ),
                    ),
                    ("cumulative_return", lambda x: (1 + x).prod() - 1),
                    (
                        "max_drawdown",
                        lambda x: (
                            (1 + x).cumprod() / (1 + x).cumprod().cummax() - 1
                        ).min(),
                    ),
                ]
            )
        else:
            performance_metrics = None

        return {
            "resilience_data": resilience,
            "resilience_periods": resilience_periods,
            "performance_metrics": performance_metrics,
            "high_threshold": high_threshold,
            "low_threshold": low_threshold,
        }
    else:
        return {"resilience_data": resilience}


def save_complex_data(data, file_path, format="json"):
    """
    Save a complex nested data structure containing pandas DataFrames, Series,
    dictionaries, floats, booleans, etc. to a file.

    Parameters:
    -----------
    data : dict
        The complex nested data structure to save
    file_path : str
        Path where the file should be saved
    format : str, optional
        Format to save the data in. Options are "json", "pickle", or "hybrid"
        Default is "json"

    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    if format == "json":
        try:
            serialized_data = convert_to_json_serializable(data)
            with open(file_path, "w") as f:
                json.dump(serialized_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving data to JSON: {e}")
            return False

    elif format == "pickle":
        try:
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Error saving data to pickle: {e}")
            return False

    elif format == "hybrid":
        # Save DataFrames and Series as separate files, create a metadata file
        try:
            base_dir = os.path.dirname(os.path.abspath(file_path))
            base_name = os.path.basename(file_path).split(".")[0]
            data_dir = os.path.join(base_dir, f"{base_name}_data")
            os.makedirs(data_dir, exist_ok=True)

            # Process the data and extract DataFrames/Series
            metadata = {}
            processed_data = extract_and_save_dataframes(data, data_dir, metadata)

            # Save the processed data as JSON
            metadata_path = os.path.join(base_dir, f"{base_name}_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Save the processed data (with references to DataFrames) as JSON
            with open(file_path, "w") as f:
                json.dump(processed_data, f, indent=2)

            return True
        except Exception as e:
            print(f"Error saving data with hybrid approach: {e}")
            return False
    else:
        print(f"Unsupported format: {format}. Use 'json', 'pickle', or 'hybrid'")
        return False


def convert_to_json_serializable(data):
    """
    Convert a nested data structure to be JSON serializable.
    Converts pandas DataFrames, Series, numpy arrays, etc.

    Parameters:
    -----------
    data : object
        The data to convert

    Returns:
    --------
    object
        JSON serializable representation of the data
    """
    if isinstance(data, dict):
        return {k: convert_to_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_json_serializable(item) for item in data]
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return {
            "_type": type(data).__name__,
            "data": data.to_dict(
                orient="records" if isinstance(data, pd.DataFrame) else "dict"
            ),
            "index": data.index.tolist(),
            "columns": (
                data.columns.tolist() if isinstance(data, pd.DataFrame) else None
            ),
        }
    elif isinstance(data, np.ndarray):
        return {
            "_type": "ndarray",
            "data": data.tolist(),
            "dtype": str(data.dtype),
            "shape": data.shape,
        }
    elif isinstance(data, (np.integer, np.floating)):
        return data.item()  # Convert numpy types to Python native types
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    else:
        return str(data)  # Convert other types to strings


def extract_and_save_dataframes(data, data_dir, metadata, path=""):
    """
    Extract DataFrames and Series from a nested structure, save them separately,
    and replace them with references in the original structure.

    Parameters:
    -----------
    data : object
        The data structure to process
    data_dir : str
        Directory to save extracted DataFrames and Series
    metadata : dict
        Metadata dictionary to update with information about saved files
    path : str, optional
        Current path in the nested structure (for reference creation)

    Returns:
    --------
    object
        Processed data structure with references
    """
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            current_path = f"{path}.{k}" if path else k
            result[k] = extract_and_save_dataframes(v, data_dir, metadata, current_path)
        return result
    elif isinstance(data, list):
        return [
            extract_and_save_dataframes(item, data_dir, metadata, f"{path}[{i}]")
            for i, item in enumerate(data)
        ]
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        # Save DataFrame/Series to CSV/Parquet
        file_name = f"{path.replace('.', '_').replace('[', '_').replace(']', '_')}"

        # Use parquet for DataFrames (better for ML), CSV for Series
        if isinstance(data, pd.DataFrame):
            file_path = os.path.join(data_dir, f"{file_name}.parquet")
            data.to_parquet(file_path)
            format_type = "parquet"
        else:  # Series
            file_path = os.path.join(data_dir, f"{file_name}.csv")
            data.to_csv(file_path)
            format_type = "csv"

        # Add reference to metadata
        metadata[path] = {
            "type": type(data).__name__,
            "file": file_path,
            "format": format_type,
            "shape": data.shape,
        }

        # Return reference
        return {"_type": "reference", "path": path}
    else:
        return convert_to_json_serializable(data)


def load_complex_data(file_path, format="json"):
    """
    Load a previously saved complex data structure.

    Parameters:
    -----------
    file_path : str
        Path to the saved file
    format : str, optional
        Format of the saved data. Options are "json", "pickle", or "hybrid"
        Default is "json"

    Returns:
    --------
    dict
        The loaded data structure
    """
    if format == "json":
        try:
            with open(file_path, "r") as f:
                serialized_data = json.load(f)
            return convert_from_json_serializable(serialized_data)
        except Exception as e:
            print(f"Error loading data from JSON: {e}")
            return None

    elif format == "pickle":
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading data from pickle: {e}")
            return None

    elif format == "hybrid":
        try:
            base_dir = os.path.dirname(os.path.abspath(file_path))
            base_name = os.path.basename(file_path).split(".")[0]
            metadata_path = os.path.join(base_dir, f"{base_name}_metadata.json")

            # Load metadata
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Load main data with references
            with open(file_path, "r") as f:
                data = json.load(f)

            # Resolve references
            return resolve_references(data, metadata)
        except Exception as e:
            print(f"Error loading data with hybrid approach: {e}")
            return None
    else:
        print(f"Unsupported format: {format}. Use 'json', 'pickle', or 'hybrid'")
        return None


def convert_from_json_serializable(data):
    """
    Convert JSON serialized data back to its original format.

    Parameters:
    -----------
    data : object
        The JSON serialized data

    Returns:
    --------
    object
        The data in its original format
    """
    if isinstance(data, dict):
        if "_type" in data:
            if data["_type"] == "DataFrame":
                df = pd.DataFrame(data["data"])
                if data["index"] is not None:
                    df.index = data["index"]
                if data["columns"] is not None:
                    df.columns = data["columns"]
                return df
            elif data["_type"] == "Series":
                return pd.Series(
                    data=list(data["data"].values()),
                    index=data["index"] if data["index"] else None,
                )
            elif data["_type"] == "ndarray":
                return np.array(data["data"], dtype=eval(data["dtype"]))
            else:
                return {k: convert_from_json_serializable(v) for k, v in data.items()}
        else:
            return {k: convert_from_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_from_json_serializable(item) for item in data]
    else:
        return data


def resolve_references(data, metadata):
    """
    Resolve references in the hybrid format by loading actual data files.

    Parameters:
    -----------
    data : object
        The data structure with references
    metadata : dict
        Metadata containing information about saved files

    Returns:
    --------
    object
        The data structure with references replaced by actual data
    """
    if isinstance(data, dict):
        if "_type" in data and data["_type"] == "reference":
            # Load the referenced file
            ref_info = metadata[data["path"]]
            file_path = ref_info["file"]

            if ref_info["format"] == "parquet":
                return pd.read_parquet(file_path)
            elif ref_info["format"] == "csv":
                if ref_info["type"] == "Series":
                    return pd.read_csv(file_path, index_col=0, squeeze=True)
                else:
                    return pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported format: {ref_info['format']}")
        else:
            return {k: resolve_references(v, metadata) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_references(item, metadata) for item in data]
    else:
        return data


def main(step):
    step = 40
    price_df = pd.read_csv(
        "nasdq_recent_20254307.csv", parse_dates=["Date"], index_col="Date"
    )
    price_df.columns = price_df.columns.str.lower()

    # filename = f"hurst_{step}D_rolling_20255924.csv"
    h_df = pd.read_csv("hurst_40D_rolling_20254307.csv", parse_dates=["Date"])
    h_df.columns = h_df.columns.str.lower()
    h_df.set_index("date", inplace=True)

    results_pat = analyze_hurst_patterns(
        h_df, price_df, h_column="hurst", window_size=step
    )
    results_reg = regime_detection_from_h(
        h_df, price_df, h_column="hurst", window_size=step, plot_results=True
    )
    result_mar = detect_market_resilience(
        h_df, price_df, h_column="hurst", window_size=step, threshold_percentile=75
    )

    save_complex_data(results_pat, f"results_pat_{step}.json", format="hybrid")
    save_complex_data(results_reg, f"results_reg_{step}.json", format="hybrid")
    save_complex_data(result_mar, f"results_mar_{step}.json", format="hybrid")


def create_regime_centric_data(h_df, price_df, step):
    # Get all your existing analyses

    filename = f"data/hurst_40D_rolling_20251527.csv"
    h_df = pd.read_csv(filename, parse_dates=["Date"])
    h_df.columns = h_df.columns.str.lower()
    h_df.set_index("date", inplace=True)

    results_pat = analyze_hurst_patterns(
        h_df, price_df, h_column="hurst", window_size=step
    )
    results_reg = regime_detection_from_h(
        h_df, price_df, h_column="hurst", window_size=step
    )
    result_mar = detect_market_resilience(
        h_df, price_df, h_column="hurst", window_size=step, threshold_percentile=75
    )

    save_complex_data(results_pat, f"results_pat_{step}.json", format="hybrid")
    save_complex_data(results_reg, f"results_reg_{step}.json", format="hybrid")
    save_complex_data(result_mar, f"results_mar_{step}.json", format="hybrid")

    hurst = align_hurst_to_prices(price_df, h_df["hurst"])
    hurst_dict = {"hurst": hurst}
    h_df = pd.DataFrame(hurst_dict, index=price_df.index)

    # Create a unified dataframe with date as index
    unified_df = pd.DataFrame(index=h_df.index)

    # Add price and hurst data
    unified_df["price"] = price_df.close.values
    unified_df["hurst"] = h_df["hurst"]

    # Add regime information
    unified_df["regime"] = None
    unified_df["regime_duration"] = None
    unified_df["regime_progress"] = None  # How far along we are in the regime (0-1)

    # Map each date to its corresponding regime
    for _, row in results_reg["duration_data"].iterrows():
        mask = (unified_df.index >= row["start_date"]) & (
            unified_df.index <= row["end_date"]
        )
        unified_df.loc[mask, "regime"] = row["regime"]
        unified_df.loc[mask, "regime_duration"] = row["duration_days"]

        # Calculate progress through the regime (0 at start, 1 at end)
        if row["duration_days"] > 0:
            date_range = pd.date_range(start=row["start_date"], end=row["end_date"])
            progress_dict = {
                date: idx / len(date_range) for idx, date in enumerate(date_range)
            }
            for date in unified_df.index[mask]:
                if date in progress_dict:
                    unified_df.loc[date, "regime_progress"] = progress_dict[date]

    unified_df["regime"] = unified_df["regime"].ffill()
    unified_df["regime_progress"] = unified_df["regime_progress"].ffill()
    unified_df["regime_duration"] = unified_df["regime_duration"].ffill()

    unified_df = extend_regimes_preserve_duration(
        unified_df=unified_df, regimes_df=results_reg["duration_data"]
    )

    # Add pattern information
    for pattern_col in results_pat.columns:
        unified_df[f"pattern_{pattern_col}"] = results_pat[pattern_col]

    # Add resilience metrics
    for res_col in result_mar.columns:
        unified_df[f"resilience_{res_col}"] = result_mar[res_col]

    # Calculate additional regime-specific metrics
    regimes = unified_df["regime"].unique()

    # Add regime duration - how long has the current regime lasted
    unified_df["regime_duration"] = (
        unified_df["regime"]
        .groupby((unified_df["regime"] != unified_df["regime"].shift(1)).cumsum())
        .cumcount()
        + 1
    )

    return unified_df


def enhance_with_regime_metrics(unified_df, h_df, price_df):
    """
    Add regime-specific metrics to the unified dataframe.
    """
    # Get list of regimes
    regimes = unified_df["regime"].unique()

    # For each regime, calculate specific statistics
    for regime in regimes:
        regime_mask = unified_df["regime"] == regime

        # Skip if no data for this regime
        if not any(regime_mask):
            continue

        # Regime-specific returns
        unified_df.loc[regime_mask, "regime_return"] = unified_df.loc[
            regime_mask, "price"
        ].pct_change()

        # Regime-specific volatility
        unified_df.loc[regime_mask, "regime_volatility"] = (
            unified_df.loc[regime_mask, "regime_return"].rolling(window=10).std()
        )

        # Regime-specific Hurst behavior
        unified_df.loc[regime_mask, "regime_hurst_mean"] = (
            unified_df.loc[regime_mask, "hurst"].rolling(window=5).mean()
        )
        unified_df.loc[regime_mask, "regime_hurst_trend"] = (
            unified_df.loc[regime_mask, "hurst"].diff().rolling(window=5).mean()
        )

        # Regime-specific drawdown
        regime_prices = unified_df.loc[regime_mask, "price"]
        if len(regime_prices) > 0:
            rolling_max = regime_prices.cummax()
            unified_df.loc[regime_mask, "regime_drawdown"] = (
                regime_prices - rolling_max
            ) / rolling_max
            unified_df.loc[regime_mask, "regime_max_drawdown"] = unified_df.loc[
                regime_mask, "regime_drawdown"
            ].cummin()

        # Regime-specific correlation between Hurst and price
        regime_start = unified_df.loc[regime_mask].index.min()
        regime_end = unified_df.loc[regime_mask].index.max()
        if regime_start is not None and regime_end is not None:
            corr = h_df.loc[regime_start:regime_end, "hurst"].corr(
                price_df.loc[regime_start:regime_end]
            )
            unified_df.loc[regime_mask, "regime_hurst_price_corr"] = corr

        # Regime transition information - days until regime change
        # First, get the end date of this regime
        regime_info_mask = unified_df["regime"] == regime
        if any(regime_info_mask):
            regime_end_date = unified_df.loc[regime_info_mask].index.max()
            # Calculate days until regime end for each date in this regime
            for date in unified_df.loc[regime_mask].index:
                if isinstance(date, pd.Timestamp) and isinstance(
                    regime_end_date, pd.Timestamp
                ):
                    days_to_end = (regime_end_date - date).days
                    unified_df.loc[date, "days_to_regime_end"] = max(0, days_to_end)

    return unified_df


def extend_regimes_preserve_duration(unified_df, regimes_df):
    """
    Forward fill regimes while preserving original duration and extending progress beyond 1.
    """
    # Forward fill the regime
    unified_df["regime"] = unified_df["regime"].ffill()

    # Get all unique regimes in the extended data
    extended_regimes = unified_df["regime"].unique()

    for regime in extended_regimes:
        if regime is None:
            continue

        # Get original regime information
        regime_info = regimes_df[regimes_df["regime"] == regime]
        if len(regime_info) == 0:
            continue

        orig_start = regime_info["start_date"].iloc[0]
        orig_end = regime_info["end_date"].iloc[0]
        orig_duration = regime_info["duration_days"].iloc[0]

        # Preserve the original duration
        regime_mask = unified_df["regime"] == regime
        unified_df.loc[regime_mask, "regime_duration"] = orig_duration

        # Calculate progress relative to the start date
        for date in unified_df.loc[regime_mask].index:
            if isinstance(date, pd.Timestamp) and isinstance(orig_start, pd.Timestamp):
                days_since_start = (date - orig_start).days
                # Progress can exceed 1.0 for extended periods
                unified_df.loc[date, "regime_progress"] = days_since_start / max(
                    1, orig_duration
                )

        # Flag extended periods
        orig_period_mask = (unified_df.index >= orig_start) & (
            unified_df.index <= orig_end
        )
        extended_mask = regime_mask & ~orig_period_mask
        unified_df.loc[extended_mask, "regime_extended"] = True

    return unified_df


def align_hurst_to_prices(prices_data, hurst_data):
    """
    Interpolate Hurst data to match the length of the prices data.

    Parameters:
    - prices_data: DataFrame or array containing price data (6000 rows)
    - hurst_data: DataFrame or array containing Hurst exponent data (740 rows)

    Returns:
    - aligned_hurst: Array of Hurst values interpolated to match prices_data length
    """
    # Create x-coordinates for both datasets
    # For prices: 0, 1, 2, ..., 5999
    prices_x = np.arange(len(prices_data))

    # For Hurst: 0, 1, 2, ..., 739
    hurst_x = np.arange(len(hurst_data))

    # Create x-coordinates for Hurst data that match the price data scale
    # This maps the Hurst indices to equivalent positions in the price data
    hurst_x_scaled = np.linspace(0, len(prices_data) - 1, len(hurst_data))

    # Create the interpolation function
    # (Assuming hurst_data is a 1D array or a column that can be accessed)
    if hasattr(hurst_data, "values"):  # If it's a DataFrame or Series
        hurst_values = hurst_data.values.flatten()
    else:  # If it's already a numpy array
        hurst_values = hurst_data

    # Create the interpolation function
    interp_func = interp1d(
        hurst_x_scaled, hurst_values, kind="linear", fill_value="extrapolate"
    )

    # Apply interpolation to get Hurst values aligned with price data
    aligned_hurst = interp_func(prices_x)

    return aligned_hurst


# # Load price data
# price_df = pd.read_csv("data/nasdaq_futures_1D_Close_20250226.csv", parse_dates=["Date"], index_col="Date")
# price_df.columns = price_df.columns.str.lower()

# filename = f'data/hurst_8D_rolling1.csv'
# h_df = pd.read_csv(filename, parse_dates=["Date"])
# h_df.columns = h_df.columns.str.lower()
# h_df.set_index('date', inplace=True)


# df = create_regime_centric_data(h_df, price_df=price_df, step=8)

# steps = [8, 10, 12, 16]

# for step in steps:
#     filename = f'data/hurst_{step}D_rolling.csv'
#     h_df = pd.read_csv(filename, parse_dates=["Date"])
#     h_df.columns = h_df.columns.str.lower()
#     h_df.set_index('date', inplace=True)


#     df = create_regime_centric_data(h_df, price_df=price_df, step=8)
# results_pat = analyze_hurst_patterns(h_df, price_df, h_column='hurst', window_size=step)
# results_reg = regime_detection_from_h(h_df, price_df, h_column='hurst', window_size=step)
# result_mar = detect_market_resilience(h_df, price_df, h_column='hurst', window_size=step, threshold_percentile=75)


if __name__ == "__main__":
    main(step=8)

price_df = pd.read_csv(
    "nasdq_recent_20254307.csv", parse_dates=["Date"], index_col="Date"
)
price_df.columns = price_df.columns.str.lower()

price_df.resample()
df_monthly_pct_change = price_df.resample("M")["close"].mean().pct_change()
df_monthly_pct_change = df_monthly_pct_change.to_frame(name="monthly_pct_change")
df_monthly_pct_change["month_end"] = df_monthly_pct_change.index
price_df["month"] = price_df.index.to_period("M")
df_monthly_pct_change["month"] = df_monthly_pct_change["month_end"].dt.to_period("M")

df_merged = pd.merge(
    price_df,
    df_monthly_pct_change[["month", "monthly_pct_change"]],
    on="month",
    how="left",
)

# 4. Clean up the temporary 'month' columns
df_merged = df_merged.drop(columns=["month", "month_end"], errors="ignore")
df_merged["return_30"] = (df_merged["close"].pct_change(30)) * 100
df_merged["monthly_pct_change"] = df_merged["monthly_pct_change"] * 100
