import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy import stats
import pickle
from statsmodels.tsa.stattools import adfuller
import ta.volatility
import ta.volume
from rolling_hurst import rolling_hurst
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import ta
from collections import Counter


class dataPrep:
    def __init__(self, n_components=4, random_state=42):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state)

    def preprocess_features(self, df, feature_cols):
        """
        Preprocess and scale the features.

        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe containing the features
        feature_cols : list
            List of column names to use as features

        Returns:
        --------
        X_scaled : numpy.ndarray
            Scaled features
        """
        self.feature_names = feature_cols
        X = df[feature_cols].copy()

        # Handle missing values
        X = X.fillna(method="ffill").fillna(method="bfill")

        # Scale features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def reduce_dimensions(self, X_scaled):
        """
        Reduce dimensions using PCA while preserving temporal structure.

        Parameters:
        -----------
        X_scaled : numpy.ndarray
            Scaled features

        Returns:
        --------
        X_reduced : numpy.ndarray
            Reduced features
        """
        # Apply PCA
        X_reduced = self.pca.fit_transform(X_scaled)

        # Print explained variance
        explained_variance = self.pca.explained_variance_ratio_
        print(f"Explained variance by component: {explained_variance}")
        print(f"Total explained variance: {sum(explained_variance):.4f}")

        return X_reduced

    def normalize_dataframe_lengths(self, dataframes, keep="tail"):
        """
        Normalize the lengths of multiple dataframes to match the smallest dataframe.

        Parameters:
        -----------
        dataframes : list of pandas.DataFrame
            List of dataframes to normalize
        keep : str, optional (default='tail')
            Specify whether to keep the 'head' or 'tail' of the dataframes
            'head' removes rows from the beginning
            'tail' removes rows from the end

        Returns:
        --------
        list of pandas.DataFrame
            Normalized dataframes with equal length

        Raises:
        -------
        ValueError: If keep is not 'head' or 'tail'
        """
        # Check if input is empty
        if not dataframes:
            return []

        # Validate keep parameter
        if keep not in ["head", "tail"]:
            raise ValueError("keep must be either 'head' or 'tail'")

        # Find the length of the smallest dataframe
        min_length = min(len(df) for df in dataframes)

        # Normalize dataframes
        normalized_dfs = []
        for df in dataframes:
            if keep == "tail":
                # Keep the last min_length rows
                normalized_df = df.tail(min_length)
            else:
                # Keep the first min_length rows
                normalized_df = df.head(min_length)

            normalized_dfs.append(normalized_df)

        return normalized_dfs

    def add_technical_indicators(self, df, hurst, price_col="close"):
        """
        Add various technical indicators to the dataframe.
        """
        original_index = df.index
        # Add returns
        df["daily_return"] = df[price_col].pct_change()
        df["log_return"] = np.log(df[price_col] / df[price_col].shift(1))

        # Add volatility (returns_volatility)
        df["returns_volatility_60"] = df["daily_return"].rolling(window=60).std()
        df["returns_volatility_10"] = df["daily_return"].rolling(window=10).std()
        df["returns_volatility_5"] = df["daily_return"].rolling(window=5).std()
        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], window=15
        )
        df["donchian_h"] = ta.volatility.donchian_channel_hband(
            df["high"], df["low"], df["close"], window=15, offset=3, fillna=True
        )
        df["donchian_l"] = ta.volatility.donchian_channel_lband(
            df["high"], df["low"], df["close"], window=15, offset=3, fillna=True
        )

        # Add moving averages
        df["sma_20"] = df[price_col].rolling(window=20).mean()
        df["sma_50"] = df[price_col].rolling(window=50).mean()
        df["sma_200"] = df[price_col].rolling(window=200).mean()
        df["ema_50"] = df[price_col].ewm(span=50, adjust=False).mean()
        df["vwap"] = ta.volume.volume_weighted_average_price(
            df["high"], df["low"], df["close"], df["volume"], window=15
        )
        df["volume_change"] = df["volume"].pct_change()
        df["volume_change"].replace([np.inf, -np.inf], 0, inplace=True)
        df = df.dropna(subset=["volume_change"])
        df["money_flow"] = ta.volume.chaikin_money_flow(
            df["high"], df["low"], df["close"], df["volume"], window=15
        )

        # Add relative strength index (RSI)
        delta = df[price_col].diff()
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        avg_gain = up.rolling(window=14).mean()
        avg_loss = abs(down.rolling(window=14).mean())
        rs = avg_gain / avg_loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Add MACD
        df["ema_12"] = df[price_col].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df[price_col].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Add Bollinger Bands
        df["bollinger_mid"] = df[price_col].rolling(window=20).mean()
        df["bollinger_std"] = df[price_col].rolling(window=20).std()
        df["bollinger_upper"] = df["bollinger_mid"] + (df["bollinger_std"] * 2)
        df["bollinger_lower"] = df["bollinger_mid"] - (df["bollinger_std"] * 2)
        df["bbands_width"] = (df["bollinger_upper"] - df["bollinger_lower"]) / df[
            "bollinger_mid"
        ]

        # Add ADX (Average Directional Index)
        high_low = (
            df["high"] - df["low"]
            if "high" in df.columns and "low" in df.columns
            else pd.Series(0, index=df.index)
        )
        high_close = (
            np.abs(df["high"] - df[price_col].shift(1))
            if "high" in df.columns
            else pd.Series(0, index=df.index)
        )
        low_close = (
            np.abs(df["low"] - df[price_col].shift(1))
            if "low" in df.columns
            else pd.Series(0, index=df.index)
        )

        tr = pd.DataFrame(index=df.index)
        tr["h-l"] = high_low
        tr["h-pc"] = high_close
        tr["l-pc"] = low_close
        tr["tr"] = tr.max(axis=1)

        # Calculate +DM and -DM
        if "high" in df.columns and "low" in df.columns:
            df["+dm"] = df["high"].diff()
            df["-dm"] = df["low"].diff() * -1
            df["+dm"] = np.where(
                (df["+dm"] > df["-dm"]) & (df["+dm"] > 0), df["+dm"], 0
            )
            df["-dm"] = np.where(
                (df["-dm"] > df["+dm"]) & (df["-dm"] > 0), df["-dm"], 0
            )
        else:
            df["+dm"] = pd.Series(0, index=df.index)
            df["-dm"] = pd.Series(0, index=df.index)

        # Calculate smoothed TR, +DM, and -DM
        df["tr_14"] = tr["tr"].rolling(window=14).sum()
        df["+dm_14"] = df["+dm"].rolling(window=14).sum()
        df["-dm_14"] = df["-dm"].rolling(window=14).sum()

        # Calculate +DI and -DI
        df["+di_14"] = 100 * (df["+dm_14"] / df["tr_14"])
        df["-di_14"] = 100 * (df["-dm_14"] / df["tr_14"])

        # Calculate DX
        df["dx"] = 100 * (
            np.abs(df["+di_14"] - df["-di_14"]) / (df["+di_14"] + df["-di_14"])
        )

        # Calculate ADX
        df["adx_14"] = df["dx"].rolling(window=14).mean()

        cols_to_diff = []
        for col in df.columns:
            series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            result = adfuller(series)
            p_value = result[1]
            # print(f"{col} - ADF p-value: {p_value:.4f}")

            if p_value > 0.05:
                cols_to_diff.append(col)

            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df = df.dropna(subset=[col])

        print("\nNon-stationary columns detected:")
        print(cols_to_diff)

        df[cols_to_diff] = df[cols_to_diff].diff()
        df = df.dropna()

        h_column = "hurst"
        # Extract Hurst exponent values
        print("Created Hurst features...")
        h_resampled = hurst[h_column].reindex(df.index, method="ffill")
        interpolated_series = h_resampled.interpolate(
            method="time", limit_direction="both"
        )
        z_threshold = 1.5
        df["hurst"] = interpolated_series.ffill().bfill()
        df["h_lag1"] = df["hurst"].shift(1)
        df["h_rolling_mean"] = df["hurst"].rolling(window=15).mean()
        df["h_rolling_std"] = df["hurst"].rolling(window=15).std()
        df["h_z_score"] = (df["hurst"] - df["h_rolling_mean"]) / df["h_rolling_std"]
        df["h_change"] = df["hurst"] - df["h_lag1"]
        df["h_change_pct"] = df["h_change"] / df["h_lag1"]
        df["h_acceleration"] = df["h_change"] - df["h_change"].shift(1)
        df["h_std"] = df["hurst"].std()
        df["extreme_h"] = (df["h_z_score"].abs() > z_threshold).astype(int)
        df["h_acceleration"] = abs(df["h_acceleration"]) / df["h_std"]

        # Rest of the function remains the same as in the previous implementation
        # Extract Hurst exponent values
        print("Data will be differences then ready for Scaler...")
        df.to_csv("prices_hurst_tech_interp_pdiff.csv")

        self.removed_rows = original_index.difference(df.index)
        self.index = df.index
        # print(df.describe().T)
        # print(df.max(numeric_only=True))  # Check column-wise max values
        # print(df.min(numeric_only=True))  # Check column-wise min values

        return df


def gmm_kfold_cv(data, max_n_components=15, n_splits=10):
    """
    Performs k-fold cross-validation to evaluate GMM models with varying numbers of components.

    Args:
        data (numpy.ndarray): The data to fit the GMM to.
        max_n_components (int): The maximum number of GMM components to consider.
        n_splits (int): The number of folds for cross-validation.
        random_state (int): Seed for random number generation for reproducibility.

    Returns:
        dict: A dictionary containing the average BIC scores for each number of components.
    """

    kf = KFold(n_splits=n_splits, shuffle=False)
    bic_scores = {}

    for n_components in range(1, max_n_components + 1):
        print(f"Trying {n_components} for K-Fold!\n")
        fold_bic_scores = []
        for train_index, test_index in kf.split(data):
            train_data, test_data = data[train_index], data[test_index]
            gmm = GaussianMixture(
                n_components=n_components,
                init_params="k-means++",
                n_init=10,
                covariance_type="diag",
                warm_start=True,
            )
            gmm.fit(train_data)
            fold_bic_scores.append(gmm.bic(test_data))
        bic_scores[n_components] = np.mean(fold_bic_scores)

    return bic_scores


# Example implementation:
# Generate some sample data (replace with your actual data)
# np.random.seed(42)
# data1 = np.random.normal(0, 1, (500, 2))
# data2 = np.random.normal(5, 1, (500, 2))
# data = np.concatenate([data1, data2])

# Perform k-fold cross-validation


def regime_detection_gmm(
    h_df,
    price_df=None,
    h_column="hurst",
    n_regimes=3,
    window_size=30,
    min_regime_duration=5,
    plot_results=True,
):
    """
    Detect market regimes based on Hurst exponent values using Gaussian Mixture Models.
    Provides probabilistic regime assignments and analyzes transitions between regimes.

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
    window_size : int
        Window size for smoothing (default: 30)
    min_regime_duration : int
        Minimum number of consecutive periods required to constitute a regime
    plot_results : bool
        If True, generate visualization plots

    Returns:
    --------
    Dictionary containing regime analysis results:
        - 'regime_assignments': Series of regime labels for each time period
        - 'regime_probabilities': DataFrame with probability of belonging to each regime
        - 'regime_stats': DataFrame with statistics for each regime
        - 'transitions': DataFrame showing regime transition probabilities
        - 'gmm_params': Parameters of the fitted GMM model
    """
    dp = dataPrep(n_components=4)
    data = dp.add_technical_indicators(price_df, h_df, price_col="close")
    data.to_csv("prices_hurst_tech_interp.csv")
    feat_scaled = dp.preprocess_features(price_df, price_df.columns.tolist())
    print("Features are ready for PCA...")
    features = dp.reduce_dimensions(feat_scaled)
    print("PCA completed!")
    data["__removed__"] = data.index.isin(dp.removed_rows)

    # Standardize data for GMM
    print("Starting K-Fold fits per number of Regimes...\n")
    # scaler = StandardScaler()
    # data_scaled = scaler.fit_transform(data)

    cv_results = gmm_kfold_cv(features)

    # Print the results
    print("Average BIC scores for each number of components:")
    for n_components, bic_score in cv_results.items():
        print(f"n_components={n_components}: BIC={bic_score:.2f}")

    # Find the optimal number of components (lowest BIC)
    optimal_n_components = min(cv_results, key=cv_results.get)
    print(f"\nOptimal number of components: {optimal_n_components}")

    # Fit the final GMM with the optimal number of components to all of the data.
    final_gmm = GaussianMixture(
        n_components=optimal_n_components,
        random_state=42,
        n_init=10,
        covariance_type="diag",
        warm_start=True,
    )
    final_gmm.fit(features)

    # Example of getting the regimes.
    regimes = final_gmm.predict(features)
    counts = Counter(regimes)
    print("length:", len(regimes), counts)
    n_regimes = len(set(regimes))

    # Example of getting the probability of each regime for each datapoint.
    regime_probabilities = final_gmm.predict_proba(features)
    # print(regime_probabilities)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(dp.scaler, f)

    with open("models/gmm_regimes_daily.pkl", "wb") as f:
        pickle.dump(final_gmm, f)

    with open("models/pca.pkl", "wb") as f:
        pickle.dump(dp.pca, f)

    print("Index from Raw Data:", len(data.index))
    print("Index from Training Data:", len(dp.index))
    print("Index from Regime Output", len(regime_probabilities))

    # Create DataFrame with probabilities
    prob_columns = [f"prob_regime_{i}" for i in range(n_regimes)]
    regime_probabilities = pd.DataFrame(
        regime_probabilities, index=price_df.index, columns=prob_columns
    )

    # Get most likely regime for each time point
    # regime_assignments_raw = gmm.predict(data_scaled)
    regime_assignments = pd.Series(regimes, index=price_df.index)

    # Apply minimum duration filter to reduce noise
    if min_regime_duration > 1:
        regime_assignments = _apply_min_duration_filter(
            regime_assignments, min_regime_duration
        )

    # Extract GMM parameters
    gmm_params = {
        "means": final_gmm.means_,  # scaler.inverse_transform()
        "variances": final_gmm.covariances_,
        "weights": final_gmm.weights_,
    }

    # Calculate regime statistics
    print(data.columns)
    print(price_df.columns)
    regime_stats = _calculate_regime_stats(
        dp,
        data,
        regime_assignments,
        regime_probabilities,
        gmm_params,
        h_column,
        price_df,
        plot_results=plot_results,
    )

    # Calculate transition matrix
    transitions = _calculate_transition_matrix(regime_assignments)

    return {
        "regime_assignments": regime_assignments,
        "regime_probabilities": regime_probabilities,
        "regime_stats": regime_stats,
        "transitions": transitions,
        "gmm_params": gmm_params,
        "data": data,
        "data_scaled": features,
    }


def _apply_min_duration_filter(regime_assignments, min_duration):
    """
    Filter out short-lived regime changes to reduce noise
    """
    filtered_regimes = regime_assignments.copy()
    regime_runs = []

    # Identify runs of the same regime
    current_regime = filtered_regimes.iloc[0]
    run_start = 0

    for i, regime in enumerate(filtered_regimes):
        if regime != current_regime:
            regime_runs.append((run_start, i - 1, current_regime))
            run_start = i
            current_regime = regime

    # Add the last run
    regime_runs.append((run_start, len(filtered_regimes) - 1, current_regime))

    # Filter out short runs
    for start, end, regime in regime_runs:
        if end - start + 1 < min_duration:
            # Find adjacent regimes to decide what to replace with
            if start > 0:
                # Use previous regime
                prev_regime = filtered_regimes.iloc[start - 1]
                filtered_regimes.iloc[start : end + 1] = prev_regime
            elif end < len(filtered_regimes) - 1:
                # Use next regime
                next_regime = filtered_regimes.iloc[end + 1]
                filtered_regimes.iloc[start : end + 1] = next_regime

    return filtered_regimes


def _calculate_regime_stats(
    dataprep,
    h_df,
    regime_assignments,
    regime_probabilities,
    gmm_params,
    h_column,
    price_df=None,
    plot_results=True,
):
    """
    Calculate statistics for each regime
    """
    # Initialize statistics DataFrame
    stats_data = []
    h_df, regime_assignments, regime_probabilities, price_df = (
        dataprep.normalize_dataframe_lengths(
            [h_df, regime_assignments, regime_probabilities, price_df], keep="tail"
        )
    )
    print(h_df.head())

    for regime in sorted(regime_assignments.unique()):
        regime_mask = regime_assignments == regime

        # Hurst exponent statistics
        try:
            h_stats = h_df.loc[regime_mask, h_column].describe()

            stats_dict = {
                "regime": regime,
                "count": h_stats["count"],
                "h_mean": h_stats["mean"],
                "h_std": h_stats["std"],
                "h_min": h_stats["min"],
                "h_max": h_stats["max"],
                "duration_pct": regime_mask.mean() * 100,
            }
        except KeyError as e:
            print(f"{h_column} column not found in {h_df.columns}")
            h_df["regimes"] = regime_assignments
            print(h_df.head())

        # Add price statistics if price_df is provided
        if price_df is not None:
            aligned_price = price_df.reindex(h_df.index)
            regime_prices = aligned_price.loc[regime_mask]

            if not regime_prices.empty:
                # Calculate returns
                returns = regime_prices.pct_change().dropna()

                if not returns.empty:
                    stats_dict.update(
                        {
                            "return_mean": returns.mean().iloc[0] * 100,
                            "return_std": returns.std().iloc[0] * 100,
                            "sharpe": (returns.mean() / returns.std()).iloc[0]
                            * np.sqrt(252),
                            "max_drawdown": _calculate_max_drawdown(regime_prices)
                            * 100,
                        }
                    )

        stats_data.append(stats_dict)

    # Generate visualizations
    if plot_results:
        _plot_gmm_results(
            h_df,
            regime_assignments,
            regime_probabilities,
            gmm_params,
            h_column,
            price_df,
        )

    return pd.DataFrame(stats_data)


def _calculate_max_drawdown(prices):
    """
    Calculate maximum drawdown for a price series
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]  # Take first column

    # Calculate the running maximum
    running_max = prices.cummax()

    # Calculate drawdown
    drawdown = (prices - running_max) / running_max

    # Return the minimum drawdown (maximum loss)
    return drawdown.min() if not drawdown.empty else 0


def _calculate_transition_matrix(regime_assignments):
    """
    Calculate transition probabilities between regimes
    """
    n_regimes = len(regime_assignments.unique())
    transitions = np.zeros((n_regimes, n_regimes))

    # Count transitions
    for i in range(len(regime_assignments) - 1):
        from_regime = regime_assignments.iloc[i]
        to_regime = regime_assignments.iloc[i + 1]
        transitions[from_regime, to_regime] += 1

    # Convert to probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    transition_probs = transitions / row_sums

    # Create DataFrame
    regime_labels = [f"Regime {i}" for i in range(n_regimes)]
    transition_df = pd.DataFrame(
        transition_probs, index=regime_labels, columns=regime_labels
    )

    return transition_df


def _plot_gmm_results(
    h_df, regime_assignments, regime_probabilities, gmm_params, h_column, price_df=None
):
    """
    Generate visualizations for GMM regime detection results
    h_df: pandas.DataFrame
        The main data used to identify the regimes. Includes FED data, OHLCV, Hurst, and tenchnical indicators
    regime_assignments: List
        A list of regime assignments from the gmm
    regime_probabilities: pandas DataFrame Series
        A column for each regime probability
    gmm_params: Dictionary
        'means': List
            From the final gmm model, the mean of each regime
        'variance': List
            the covariances for each regime and each element in the input
        'weights': List
            the weights for each regime
    h_column: str
        Identifying the column that has the hurst exponent
    price_df: pandas.DataFrame
        Original Price information (OHLCV), but may include other columns (FED, hurst, etc.)

    """
    print(h_df.columns.tolist())
    n_regimes = len(regime_assignments.unique())
    fig = plt.figure(figsize=(15, 12))
    # h_df.index = pd.to_datetime(h_df.index)

    # Set up color map
    colors = plt.cm.tab10(np.linspace(0, 1, n_regimes))
    cmap = ListedColormap(colors)

    # Plot 1: Hurst exponent with regime coloring
    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)

    for regime in range(n_regimes):
        mask = regime_assignments == regime
        dates = h_df.index[mask]
        values = h_df.loc[mask, h_column].values
        ax1.scatter(
            dates,
            values,
            c=[colors[regime]],
            label=f"Regime {regime}",
            alpha=0.7,
            s=10,
        )

    ax1.set_title("Hurst Exponent with Regime Classification")
    ax1.set_ylabel("Hurst Exponent")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Regime probabilities stacked area chart
    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    regime_probabilities.plot.area(ax=ax2, colormap=cmap, alpha=0.7)
    ax2.set_title("Regime Probability Distribution Over Time")
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.grid(True, alpha=0.3)

    # Plot 3: GMM distribution
    ax3 = plt.subplot2grid((3, 2), (2, 0))

    # Create a range of x values
    # x = np.linspace(
    #     h_df[h_column].min() - 0.1, h_df[h_column].max() + 0.1, 1000
    # ).reshape(-1, 1)
    x_pca = np.linspace(-10, 10, 1000)

    # Plot histogram of Hurst values
    # ax3.hist(h_df[h_column], bins=30, density=True, alpha=0.5, color="gray")

    # Plot GMM components
    for i in range(n_regimes):
        mean = gmm_params["means"][i][0]
        var = gmm_params["variances"][i][0]
        weight = gmm_params["weights"][i]

        print(f"Regime {i}:")
        print(f"  Mean: {mean}")
        print(f"  Variance: {var}")
        print(f"  Weight: {weight}")

        y = (
            weight
            * (1 / np.sqrt(2 * np.pi * var))
            * np.exp(-0.5 * ((x_pca - mean) ** 2) / var)
        )
        ax3.plot(
            x_pca,
            y,
            color=colors[i],
            lw=2,
            label=f"Regime {i} (μ={mean:.2f}, σ²={var:.3f})",
        )

    ax3.set_title("GMM Components (PCA Space)")
    ax3.set_xlabel("First PCA Component")
    ax3.set_ylabel("Density")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Transition matrix heatmap
    ax4 = plt.subplot2grid((3, 2), (2, 1))
    transitions = _calculate_transition_matrix(regime_assignments)
    sns.heatmap(
        transitions,
        annot=True,
        cmap="Blues",
        ax=ax4,
        fmt=".2f",
        xticklabels=[f"R{i}" for i in range(n_regimes)],
        yticklabels=[f"R{i}" for i in range(n_regimes)],
    )
    ax4.set_title("Regime Transition Probabilities")
    ax4.set_xlabel("To Regime")
    ax4.set_ylabel("From Regime")

    # Add price overlay if provided
    if price_df is not None:
        ax1_twin = ax1.twinx()
        aligned_price = price_df.reindex(h_df.index)
        ax1_twin.plot(
            aligned_price.index,
            aligned_price.iloc[:, 0],
            color="black",
            alpha=0.3,
            lw=1,
        )
        ax1_twin.set_ylabel("Price", color="darkgray")

    plt.tight_layout()
    plt.show()

    # Additional plot for regime characteristics
    _plot_regime_characteristics(h_df, regime_assignments, h_column, price_df)


def _plot_regime_characteristics(h_df, regime_assignments, h_column, price_df=None):
    """
    Plot characteristics of each regime
    """
    n_regimes = len(regime_assignments.unique())

    # Setup plot
    fig, axes = plt.subplots(1, n_regimes, figsize=(15, 4))
    if n_regimes == 1:
        axes = [axes]

    # Set up color map
    colors = plt.cm.tab10(np.linspace(0, 1, n_regimes))

    # For each regime
    for i, regime in enumerate(sorted(regime_assignments.unique())):
        regime_mask = regime_assignments == regime

        # Get Hurst values for this regime
        h_values = h_df.loc[regime_mask, h_column]

        # Plot Hurst distribution
        axes[i].hist(h_values, bins=15, alpha=0.7, color=colors[i])
        axes[i].set_title(f"Regime {regime} (n={len(h_values)})")
        axes[i].set_xlabel("Hurst Exponent")
        axes[i].grid(True, alpha=0.3)

        # Add statistics as text
        stats_text = f"Mean: {h_values.mean():.3f}\nStd: {h_values.std():.3f}"

        # Add price return statistics if available
        if price_df is not None:
            aligned_price = price_df.reindex(h_df.index)
            regime_prices = aligned_price.loc[regime_mask]

            if not regime_prices.empty:
                returns = regime_prices.pct_change().dropna()

                if not returns.empty:
                    stats_text += f"\nRet: {returns.mean().iloc[0]*100:.2f}%"
                    stats_text += f"\nVol: {returns.std().iloc[0]*100:.2f}%"
                    stats_text += f"\nSharpe: {(returns.mean()/returns.std()).iloc[0]*np.sqrt(252):.2f}"

        axes[i].text(
            0.05,
            0.95,
            stats_text,
            transform=axes[i].transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=0.1),
        )

    plt.tight_layout()
    plt.show()


def analyze_regimes(df):
    """
    Generate comprehensive statistics for each regime.
    """
    # Create regime change indicator
    df["regime_change"] = df["regime"].diff().fillna(0).astype(int)

    # Create regime duration
    df["regime_duration"] = 0
    current_duration = 0
    current_regime = None

    for i, row in df.iterrows():
        if pd.isna(row["regime"]):
            current_duration = 0
            current_regime = None
        elif row["regime"] != current_regime:
            current_duration = 1
            current_regime = row["regime"]
        else:
            current_duration += 1

        df.at[i, "regime_duration"] = current_duration

    # Identify regime start and end dates
    regime_changes = df[df["regime_change"] == 1].index.tolist()
    regime_changes = [df.index[0]] + regime_changes + [df.index[-1]]

    regime_periods = []
    for i in range(len(regime_changes) - 1):
        start_date = regime_changes[i]
        end_date = regime_changes[i + 1]
        regime_data = df.loc[start_date:end_date]

        if len(regime_data) > 0:
            regime = regime_data["regime"].iloc[0]
            regime_name = regime_data["regime_name"].iloc[0]

            period_info = {
                "regime": regime,
                "regime_name": regime_name,
                "start_date": start_date,
                "end_date": end_date,
                "duration": len(regime_data),
                "return": (
                    regime_data["close"].iloc[-1] / regime_data["close"].iloc[0] - 1
                )
                * 100,
            }

            regime_periods.append(period_info)

    # Generate statistics for each regime
    regime_stats = pd.DataFrame()

    for regime in df["regime"].dropna().unique():
        regime_data = df[df["regime"] == regime]
        regime_name = (
            regime_data["regime_name"].iloc[0]
            if len(regime_data) > 0
            else f"Regime {regime}"
        )

        # Skip if not enough data
        if len(regime_data) < 20:
            continue

        stats_dict = {
            "regime": regime,
            "regime_name": regime_name,
            "count": len(regime_data),
            "pct_of_total": len(regime_data) / len(df) * 100,
            "avg_duration": regime_data["regime_duration"].mean(),
            "max_duration": regime_data["regime_duration"].max(),
            "min_duration": regime_data["regime_duration"].min(),
            "occurrences": sum(df["regime_change"] & (df["regime"] == regime)),
            # Return statistics
            "mean_return": regime_data["daily_return"].mean() * 100,  # as percentage
            "median_return": regime_data["daily_return"].median()
            * 100,  # as percentage
            "std_return": regime_data["daily_return"].std() * 100,  # as percentage
            "skew_return": regime_data["daily_return"].skew(),
            "kurtosis_return": regime_data["daily_return"].kurtosis(),
            "positive_days_pct": (regime_data["daily_return"] > 0).mean() * 100,
            "cumulative_return": (np.cumprod(1 + regime_data["daily_return"]) - 1).iloc[
                -1
            ]
            * 100,  # as percentage
            "sharpe_ratio": (
                regime_data["daily_return"].mean() / regime_data["daily_return"].std()
            )
            * np.sqrt(252),
            "max_drawdown": (
                (regime_data["close"] / regime_data["close"].cummax()) - 1
            ).min()
            * 100,  # as percentage
            # Volatility statistics
            "mean_volatility": regime_data["returns_volatility_10"].mean()
            * 100,  # as percentage
            "median_volatility": regime_data["returns_volatility_10"].median()
            * 100,  # as percentage
            "min_volatility": regime_data["returns_volatility_10"].min()
            * 100,  # as percentage
            "max_volatility": regime_data["returns_volatility_10"].max()
            * 100,  # as percentage
            "annualized_volatility": regime_data["returns_volatility_10"].mean()
            * np.sqrt(252)
            * 100,  # as percentage
            # Hurst exponent statistics
            "mean_hurst": regime_data["hurst"].mean(),
            "median_hurst": regime_data["hurst"].median(),
            "min_hurst": regime_data["hurst"].min(),
            "max_hurst": regime_data["hurst"].max(),
            "std_hurst": regime_data["hurst"].std(),
            # Technical indicator statistics
            "mean_rsi": regime_data["rsi_14"].mean(),
            "median_rsi": regime_data["rsi_14"].median(),
            "rsi_above_70_pct": (regime_data["rsi_14"] > 70).mean() * 100,
            "rsi_below_30_pct": (regime_data["rsi_14"] < 30).mean() * 100,
            "mean_adx": regime_data["adx_14"].mean(),
            "median_adx": regime_data["adx_14"].median(),
            "adx_above_25_pct": (regime_data["adx_14"] > 25).mean() * 100,
            "mean_macd": regime_data["macd"].mean(),
            "mean_macd_hist": regime_data["macd_hist"].mean(),
            "macd_positive_pct": (regime_data["macd"] > 0).mean() * 100,
            "macd_hist_positive_pct": (regime_data["macd_hist"] > 0).mean() * 100,
            # Bollinger Bands statistics
            "mean_bbands_width": regime_data["bbands_width"].mean(),
            "median_bbands_width": regime_data["bbands_width"].median(),
            "price_above_upper_bb_pct": (
                regime_data["close"] > regime_data["bollinger_upper"]
            ).mean()
            * 100,
            "price_below_lower_bb_pct": (
                regime_data["close"] < regime_data["bollinger_lower"]
            ).mean()
            * 100,
            "price_within_bb_pct": (
                (regime_data["close"] <= regime_data["bollinger_upper"])
                & (regime_data["close"] >= regime_data["bollinger_lower"])
            ).mean()
            * 100,
            # Trend statistics
            "price_above_ema50_pct": (
                regime_data["close"] > regime_data["ema_50"]
            ).mean()
            * 100,
            "price_above_sma200_pct": (
                regime_data["close"] > regime_data["sma_200"]
            ).mean()
            * 100,
            "sma20_above_sma50_pct": (
                regime_data["sma_20"] > regime_data["sma_50"]
            ).mean()
            * 100,
            # Distribution statistics
            "return_distribution_normal_pvalue": stats.normaltest(
                regime_data["daily_return"].dropna()
            )[1],
            "volatility_distribution_normal_pvalue": stats.normaltest(
                regime_data["returns_volatility_10"].dropna()
            )[1],
            # Price movement
            "avg_price": regime_data["close"].mean(),
            "min_price": regime_data["close"].min(),
            "max_price": regime_data["close"].max(),
            "price_range_pct": (regime_data["close"].max() - regime_data["close"].min())
            / regime_data["close"].min()
            * 100,
        }

        # Add autocorrelation statistics for returns at different lags
        for lag in [1, 5, 10, 20]:
            stats_dict[f"autocorr_returns_lag{lag}"] = regime_data[
                "daily_return"
            ].autocorr(lag)

        # Calculate transitions into other regimes
        next_regimes = df.loc[regime_data.index]["regime"].shift(-1).dropna()
        transitions = next_regimes[next_regimes != regime].value_counts(normalize=True)

        for next_regime, prob in transitions.items():
            next_regime_name = (
                df[df["regime"] == next_regime]["regime_name"].iloc[0]
                if len(df[df["regime"] == next_regime]) > 0
                else f"Regime {next_regime}"
            )
            stats_dict[f"transition_to_{next_regime_name}"] = prob * 100

        # Add to regime stats dataframe
        regime_stats = pd.concat(
            [regime_stats, pd.DataFrame([stats_dict])], ignore_index=True
        )

    # Create a dataframe of regime periods
    regime_periods_df = pd.DataFrame(regime_periods)

    return regime_stats, regime_periods_df


def calculate_gmm_bic(X, gmm):
    """
    Calculate the Bayesian Information Criterion (BIC) for a Gaussian Mixture Model.

    Parameters:
    -----------
    X : numpy.ndarray
        The input data array of shape (n_samples, n_features)
    gmm : sklearn.mixture.GaussianMixture
        A fitted Gaussian Mixture Model

    Returns:
    --------
    float
        The Bayesian Information Criterion (BIC) value

    Notes:
    ------
    BIC = k * ln(n) - 2 * log-likelihood
    where:
    - k is the number of parameters
    - n is the number of samples
    - log-likelihood is calculated from the model
    """
    # Number of samples
    n = X.shape[0]

    # Number of features
    d = X.shape[1]

    # Number of components in the GMM
    n_components = gmm.n_components

    # Calculate number of parameters
    # For each component: mean (d parameters), covariance (d*(d+1)/2 parameters), and weight (1 parameter)
    # Total parameters per component = d + d*(d+1)/2 + 1
    param_per_component = d + (d * (d + 1) // 2) + 1
    k = n_components * param_per_component

    # Calculate log-likelihood
    log_likelihood = np.sum(gmm.score_samples(X))

    # Calculate BIC
    bic = k * np.log(n) - 2 * log_likelihood

    return bic


def main():
    step = 8
    price_df = pd.read_csv(
        "data/processed/processed_data_gaus.csv",
        parse_dates=["date"],
        index_col="date",
    )
    price_df.columns = price_df.columns.str.lower()
    if price_df.columns[0] == "close":
        pass
    else:
        column_name = price_df.columns[0]
        price_df.drop(column_name, axis=1, inplace=True)

    # price_df = price_df.loc[price_df.index >= "2018-01-01"]

    # h_df = rolling_hurst(
    #     price_df["close"], window_size=63, step=step, min_rs_window=5, max_rs_window=10
    # )

    h_df = pd.read_csv("data/hurst_40D_rolling_20252127.csv")
    h_df.drop(h_df.columns[0], axis=1, inplace=True)

    # filename = f"hurst_{step}D_rolling1.csv"
    # h_df = pd.read_csv(filename, parse_dates=["Date"])
    h_df.columns = h_df.columns.str.lower()
    h_df["date"] = pd.to_datetime(h_df["date"])
    h_df.set_index("date", inplace=True)
    # h_df = h_df.loc[h_df.index >= "2018-10-01"]

    results = regime_detection_gmm(
        h_df,
        price_df,
        h_column="hurst",
        window_size=16,
        n_regimes=6,
        plot_results=True,
    )

    data = results["data"]
    regimes = results["regime_assignments"]
    data["regime"] = regimes
    data["regime_name"] = regimes

    regime_analysis = analyze_regimes(data)

    return results, regime_analysis


if __name__ == "__main__":
    results, regime_analysis = main()
    print(regime_analysis)
    import pickle

    with open("regime_det_gmm.pkl", "wb") as f:
        pickle.dump(results, f)
    with open("regime_gmm_analysis.pkl", "wb") as f:
        pickle.dump(regime_analysis, f)


import pickle
import pandas as pd

file = "regime_det_gmm.pkl"

with open(file, "rb") as f:
    res = pickle.load(f)
import pandas as pd

pd.read_csv("data/processed/processed_data_gaus.csv")
