import numpy as np
import pandas as pd
import itertools
from rolling_hurst import rolling_hurst


class HurstParameterOptimizer:
    def __init__(self, data, hurst_calculation_func):
        """
        Initialize the optimizer with data and Hurst calculation function

        :param data: DataFrame containing price and returns data
        :param hurst_calculation_func: Function to calculate Hurst exponent
        """
        self.data = data
        self.hurst_func = hurst_calculation_func

        self.best_params = None
        self.best_score = float(0)

        # Tracking optimization history
        self.optimization_history = []

    def compute_objective_score(self, hurst_values, returns):
        """
        Compute a composite score based on Hurst values and returns

        :param hurst_values: Array of Hurst exponent values
        :param returns: Array of returns data
        :return: Composite score
        """
        returns = np.asarray(returns)
        returns = returns[~np.isnan(returns)]

        # Handle potential length mismatches
        min_length = min(len(hurst_values), len(returns))
        hurst_values = hurst_values[:min_length]
        returns = returns[:min_length]

        # Remove NaNs
        returns = returns[~np.isnan(returns)]
        hurst_values = hurst_values[~np.isnan(returns)]
        # valid_mask = ~np.isnan(hurst_values)
        # hurst_values = hurst_values[valid_mask]
        # returns = returns[valid_mask]

        # Extensive debugging print
        # print("Debugging Score Calculation:")
        # print(f"Hurst values length: {len(hurst_values)}")
        # print(f"Returns length: {len(returns)}")
        # print("RETURNS: ", returns[:10])

        # constraint
        out_of_range_count = np.sum((hurst_values < 0) | (hurst_values > 1))

        # Require at least 3 values for meaningful correlation
        if len(hurst_values) < 3:
            print("Insufficient data for correlation")
            return float("-inf")

        try:
            # Correlation with future returns (shift returns by 1)
            future_returns = returns[1:]
            hurst_for_future = hurst_values[:-1]

            future_returns_corr = np.corrcoef(hurst_for_future, future_returns)[0, 1]

            # Use absolute returns as volatility proxy
            try:
                # Ensure lengths match for correlation
                min_length = min(len(hurst_values), len(returns))
                hurst_values_for_corr = hurst_values[:min_length]
                abs_returns = np.abs(returns[:min_length])

                # Check for constant values which can cause correlation issues
                if np.std(hurst_values_for_corr) == 0 or np.std(abs_returns) == 0:
                    print("Warning: Constant values prevent correlation calculation")
                    volatility_corr = 0
                else:
                    # Calculate correlation
                    volatility_corr = self.safe_correlation(
                        hurst_values_for_corr, abs_returns
                    )
                    # Additional debugging
                # print(
                #     f"Hurst values stats: mean={np.mean(hurst_values_for_corr)}, std={np.std(hurst_values_for_corr)}"
                # )
                # print(
                #     f"Abs returns stats: mean={np.mean(abs_returns)}, std={np.std(abs_returns)}"
                # )
                # print(f"Volatility Correlation: {volatility_corr}")
            except Exception as e:
                print(f"Error in volatility correlation calculation: {e}")
                volatility_corr = 0

            # Print intermediate correlation values
            # print(f"Future Returns Correlation: {future_returns_corr}")
            # print(f"Volatility Correlation: {volatility_corr}")

            # Information gain (proxy using correlation magnitude)
            info_gain = np.abs(future_returns_corr)

            # Complexity penalty to prevent overfitting
            complexity_penalty = 1 / (1 + len(hurst_values))

            # Composite score with more robust calculation
            composite_score = (
                0.5 * np.abs(future_returns_corr)
                + 0.2 * np.abs(volatility_corr)
                + 0.2 * info_gain
                + 0.1 * complexity_penalty
                - out_of_range_count
            )

            # Additional debugging
            print(f"Composite Score: {composite_score}")

            # Check for NaN or infinite values
            if np.isnan(composite_score) or np.isinf(composite_score):
                print("Warning: Score is NaN or Inf")
                return float("-inf")

            return composite_score

        except Exception as e:
            print(f"Error in score calculation: {e}")
            return float("-inf")

    def generate_parameter_grid(self, param_ranges):
        """
        Generate a grid of parameter combinations

        :param param_ranges: Dictionary of parameter ranges
        :return: List of parameter combinations
        """
        param_names = ["window_size", "step", "min_rs_window", "max_rs_window"]
        param_values = [param_ranges[name] for name in param_names]

        return [
            dict(zip(param_names, combo)) for combo in itertools.product(*param_values)
        ]

    def bayesian_optimization(self, param_ranges, n_iterations=5):
        """
        Perform Bayesian-inspired parameter optimization

        :param param_ranges: Dictionary of parameter ranges to explore
        :param n_iterations: Number of optimization iterations
        """
        # Generate initial parameter grid
        parameter_grid = self.generate_parameter_grid(param_ranges)

        # Shuffle to avoid systematic bias
        np.random.shuffle(parameter_grid)

        # Limit iterations to grid size if smaller
        n_iterations = min(n_iterations, len(parameter_grid))

        for params in parameter_grid[:n_iterations]:
            try:
                # Print detailed parameter information
                # print("\n--- Parameter Set ---")
                # print(f"Parameters: {params}")
                # print(f"Total data length: {len(self.data)}")

                # Calculate Hurst values using the provided function
                hurst_values = self.hurst_func(
                    self.data["Close"],
                    window_size=params["window_size"],
                    step=params["step"],
                    min_rs_window=params["min_rs_window"],
                    max_rs_window=params["max_rs_window"],
                )

                # Diagnostic print for Hurst values
                # print(f"Hurst values length: {len(hurst_values)}")
                # print(f"Hurst values first few: {hurst_values[:5]}")

                # Diagnostic print for returns
                # print(f"Returns length: {len(self.data['returns'])}")
                # print(f"Returns first few: {self.data['returns'][:5]}")

                # Compute objective score
                score = self.compute_objective_score(hurst_values, self.data["returns"])

                # Track optimization history
                self.optimization_history.append(
                    {
                        "params": params,
                        "score": score,
                        "hurst_values_length": len(hurst_values),
                    }
                )

                # Update best parameters
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    print(f"New best parameters found:")
                    print(f"Parameters: {params}")
                    print(f"Score: {score}")
                    print(f"Hurst Values Length: {len(hurst_values)}\n")

            except Exception as e:
                print(f"Error processing parameters {params}: {e}")

        return self.best_params, self.best_score

    def neighborhood_search(self, base_params, neighborhood_size=2):
        """
        Perform local neighborhood search around best parameters

        :param base_params: Base parameters to explore around
        :param neighborhood_size: Size of parameter neighborhood to explore
        """
        # Define parameter perturbation ranges
        param_deltas = {
            "window_size": list(range(-neighborhood_size, neighborhood_size + 1)),
            "step": list(range(-neighborhood_size, neighborhood_size + 1)),
            "min_rs_window": list(range(-neighborhood_size, neighborhood_size + 1)),
            "max_rs_window": list(range(-neighborhood_size, neighborhood_size + 1)),
        }

        # Generate neighborhood parameter combinations
        neighborhood_params = []
        for param, deltas in param_deltas.items():
            for delta in deltas:
                new_params = base_params.copy()
                new_params[param] += delta

                # Ensure parameters remain positive and sensible
                new_params[param] = max(1, new_params[param])

                neighborhood_params.append(new_params)

        # Evaluate neighborhood parameters
        for params in neighborhood_params:
            try:
                hurst_values = self.hurst_func(
                    self.data["Close"],
                    window_size=params["window_size"],
                    step=params["step"],
                    min_rs_window=params["min_rs_window"],
                    max_rs_window=params["max_rs_window"],
                )

                score = self.compute_objective_score(hurst_values, self.data["returns"])

                # Update best parameters if improved
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    print(f"Improved parameters in neighborhood search:")
                    print(f"Parameters: {params}")
                    print(f"Score: {score}\n")
                    print(f"Hurst Values Length: {len(hurst_values)}\n")

            except Exception as e:
                print(f"Error processing neighborhood parameters {params}: {e}")

        return self.best_params, self.best_score

    def safe_correlation(self, x, y):
        # print("\nDetailed Correlation Diagnostics:")
        # print(f"x length: {len(x)}, y length: {len(y)}")
        # print(
        #     f"x unique values: {len(np.unique(x))}, y unique values: {len(np.unique(y))}"
        # )
        # print(f"x values: {x[:10]}")
        # print(f"y values: {y[:10]}")

        # Ensure arrays are float type
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        # Remove any infinities
        mask = ~(np.isinf(x) | np.isinf(y))
        x = x[mask]
        y = y[mask]

        # Compute correlation manually to diagnose issues
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)

        numerator = np.sum(x_centered * y_centered)
        denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))

        # print(f"Numerator: {numerator}")
        # print(f"Denominator: {denominator}")

        if denominator == 0:
            print("Denominator is zero, cannot compute correlation")
            return 0

        correlation = numerator / denominator

        # print(f"Manual Correlation computed: {correlation}")
        return correlation


# In your correlation calculation
# volatility_corr = safe_correlation(hurst_values_for_corr, abs_returns)


def example_hurst_calculation(
    price_data, window_size, step, min_rs_window, max_rs_window
):
    """
    Placeholder Hurst calculation function
    Replace with your actual Hurst calculation method

    :return: Array of Hurst exponent values
    """
    # Simulate multiple Hurst values based on window and step
    hurst = rolling_hurst(price_data, window_size, step, min_rs_window, max_rs_window)
    hurst = hurst["Hurst"].values
    return hurst


def main():
    # Prepare example data
    data = pd.read_csv("nasdaq_futures_1D_Close_20250324.csv", index_col="Date")
    data.drop(data.columns[0], axis=1, inplace=True)
    data["returns"] = data["Close"].pct_change()
    # data["volatility"] = (data["returns"].mean() * np.sqrt(252) * 100,)  # as percentage
    # hurst = pd.read_csv("hurst_8D_rolling_20255924.csv", index_col="Date")
    # hurst.drop(hurst.columns[0], axis=1, inplace=True)
    # data["returns"].values[1:]
    # abs_returns = np.abs(data["returns"].values[1:])

    # volatility_corr = np.corrcoef(hurst["Hurst"][1:20], abs_returns[1:16])[0, 1]

    np.random.seed(42)
    # data = pd.DataFrame(
    #     {
    #         "price": np.cumsum(np.random.normal(0, 0.1, 6200)),
    #         "returns": np.random.normal(0, 0.01, 6200),
    #         "volatility": np.abs(np.random.normal(0, 0.1, 6200)),
    #     }
    # )

    # Define parameter ranges to explore
    param_ranges = {
        "window_size": list(range(18, 252, 14)),  # From 20 to 200 with step 20
        "step": list(range(2, 50, 2)),  # From 5 to 50 with step 5
        "min_rs_window": list(range(10, 100, 10)),  # From 10 to 100 with step 10
        "max_rs_window": list(range(18, 252, 14)),  # From 50 to 500 with step 50
    }

    # Create optimizer
    optimizer = HurstParameterOptimizer(data, example_hurst_calculation)

    # Run Bayesian-inspired optimization
    best_params, best_score = optimizer.bayesian_optimization(
        param_ranges, n_iterations=100
    )

    print("\nBest Parameters from Bayesian Optimization:")
    print(best_params)
    print(f"Best Score: {best_score}")

    # Perform neighborhood search on best parameters
    best_params, best_score = optimizer.neighborhood_search(
        best_params, neighborhood_size=5
    )

    print("\nBest Parameters after Neighborhood Search:")
    print(best_params)
    print(f"Best Score: {best_score}")

    # Optional: Analyze optimization history
    print("\nTop 5 Parameter Configurations:")
    for entry in optimizer.optimization_history[:5]:
        print(f"Parameters: {entry['params']}, Score: {entry['score']}")


if __name__ == "__main__":
    main()
