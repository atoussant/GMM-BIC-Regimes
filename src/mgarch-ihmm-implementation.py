import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import pymc as pm
import arviz as az
import torch
from pymc.distributions.timeseries import GaussianRandomWalk
from scipy.interpolate import interp1d
from rolling_hurst import rolling_hurst
import pytensor.tensor as pt
from sklearn.preprocessing import StandardScaler
import os
import sys
import logging
from time import time
import pytensor
from theano import shared, function, config
import pickle

sys.setrecursionlimit(10000)

# Suppress warnings
warnings.filterwarnings("ignore")


def preprocess_financial_data(df):
    """
    Preprocess financial data for MGARCH-IHMM modeling

    Parameters:
    df (pd.DataFrame): DataFrame with OHLCV, RSI, MACD, and Hurst exponent

    Returns:
    pd.DataFrame: Preprocessed data for modeling
    """
    # Calculate returns if not already present
    if "returns" not in df.columns:
        df["returns"] = df["close"].pct_change().fillna(0)

    # Extract relevant features for regime detection
    features = ["returns", "volume", "rsi", "macd", "hurst"]
    model_df = df[features].copy()

    # Handle missing values
    model_df = model_df.fillna(method="ffill").fillna(0)

    # Standardize features for better mixing in MCMC
    scaler = StandardScaler()
    model_df_scaled = pd.DataFrame(
        scaler.fit_transform(model_df), index=model_df.index, columns=model_df.columns
    )

    return model_df_scaled, scaler


# def build_mgarch_ihmm_model(data, n_regimes_max=10):
#     """
#     Build a Multivariate GARCH model with Infinite Hidden Markov Model

#     Parameters:
#     data (pd.DataFrame): Preprocessed financial data
#     n_regimes_max (int): Maximum number of regimes to consider

#     Returns:
#     pm.Model: PyMC model object
#     """
#     n_features = data.shape[1]
#     n_observations = len(data)

#     with pm.Model() as model:
#         # Dirichlet Process concentration parameter (controls how many regimes we expect)
#         alpha = pm.Gamma("alpha", alpha=2.0, beta=1.0)

#         # Stick-breaking process for IHMM
#         beta = pm.Beta("beta", alpha=1.0, beta=alpha, shape=n_regimes_max - 1)
#         w = pm.Deterministic("w", beta * pt.concatenate([[1], pt.cumprod(1 - beta)]))

#         # Regime-specific parameters
#         mu = pm.Normal("mu", mu=0, sigma=1, shape=(n_regimes_max, n_features))

#         # GARCH parameters for each regime
#         # Note: We use Half-Normal for positive constraints on GARCH parameters
#         omega = pm.HalfNormal("omega", sigma=0.2, shape=(n_regimes_max, n_features))
#         alpha_garch = pm.HalfNormal(
#             "alpha_garch", sigma=0.2, shape=(n_regimes_max, n_features)
#         )
#         beta_garch = pm.HalfNormal(
#             "beta_garch", sigma=0.5, shape=(n_regimes_max, n_features)
#         )

#         # Correlation matrices for each regime (using LKJ distribution)
#         chol_packed = []
#         correlation = []
#         for k in range(n_regimes_max):
#             chol, corr, chol_stds = pm.LKJCholeskyCov(
#                 f"chol_packed_{k}",
#                 n=n_features,
#                 eta=2.0,
#                 sd_dist=pm.HalfCauchy.dist(1.0),
#             )
#             chol_packed.append(chol)

#             L = pm.expand_packed_triangular(n_features, chol_packed[k].ravel())

#             # Compute correlation matrix using matmul (@ operator in PyTensor)
#             # Use transpose explicitly since L.T isn't valid
#             corr_matrix = L @ pt.transpose(L)

#             # Store as deterministic variable
#             correlation.append(pm.Deterministic(f"correlation_{k}", corr_matrix))

#         # Initial state probabilities (stationary distribution)
#         initial_probs = pm.Dirichlet("initial_probs", a=np.ones(n_regimes_max))

#         # Transition matrix with sticky property (encourage staying in same regime)
#         kappa = pm.Gamma("kappa", alpha=50, beta=10)  # Stickiness parameter

#         # Create transition matrix with diagonal stickiness
#         remaining_stick = 1 - pt.sum(w)
#         w_full = pt.concatenate([w, [remaining_stick]])
#         transition_base = pt.tile(w_full, (n_regimes_max, 1))

#         diag_mask = pt.eye(n_regimes_max)
#         transition_probs = pm.Deterministic(
#             "transition_probs",
#             transition_base * (1 - kappa * diag_mask) + diag_mask * kappa,
#         )

#         # Manual implementation of Markov chain using Categorical distributions

#         states = pm.Categorical("states_0", p=initial_probs, shape=1)

#         for t in range(1, n_observations):
#             # The probability of the next state depends on the previous state
#             state_probs = transition_probs[states[t - 1]]
#             states = pm.Categorical(f"states_{t}", p=state_probs, shape=1)

#             # You may need to combine these later for analysis
#             if t == 1:
#                 all_states = states.reshape((1,))
#             else:
#                 all_states = pt.concatenate([all_states, states])

#         all_states = pm.Deterministic("states", all_states)

#         # PyTorch implementation of GARCH volatility process
#         def calc_garch_volatility(
#             data_tensor, states_tensor, omega_tensor, alpha_tensor, beta_tensor
#         ):
#             """Calculate GARCH volatility using PyTorch"""
#             n_obs = data_tensor.shape[0]
#             n_feat = data_tensor.shape[1]
#             h = torch.zeros((n_obs, n_feat))

#             # Initialize with empirical variance
#             h[0] = torch.mean(data_tensor**2, dim=0)

#             for t in range(1, n_obs):
#                 regime = states_tensor[t - 1]
#                 h[t] = (
#                     omega_tensor[regime]
#                     + alpha_tensor[regime] * data_tensor[t - 1] ** 2
#                     + beta_tensor[regime] * h[t - 1]
#                 )

#             return h

#         states_array = (
#             pm.sample_posterior_predictive(["states"])
#             .mean(dim=["chain", "draw"])
#             .values
#         )
#         states_tensor = torch.tensor(states_array, dtype=torch.long)

#         # Convert numpy arrays to PyTorch tensors
#         data_tensor = torch.tensor(data.values, dtype=torch.float32)
#         # states_tensor = torch.tensor(all_states, dtype=torch.long)
#         omega_tensor = torch.tensor(omega.eval(), dtype=torch.float32)
#         alpha_tensor = torch.tensor(alpha_garch.eval(), dtype=torch.float32)
#         beta_tensor = torch.tensor(beta_garch.eval(), dtype=torch.float32)

#         # Calculate GARCH volatilities
#         h_tensor = calc_garch_volatility(
#             data_tensor, states_tensor, omega_tensor, alpha_tensor, beta_tensor
#         )
#         h = h_tensor.numpy()

#         # Data likelihood
#         for t in range(n_observations):
#             regime = states[t]
#             sigma_t = pm.math.sqrt(h[t])
#             cov_t = pt.diag(sigma_t) @ correlation[regime] @ pt.diag(sigma_t)
#             pm.MvNormal(f"obs_{t}", mu=mu[regime], cov=cov_t, observed=data.values[t])

#     return model

# this function runs
# def build_mgarch_ihmm_model(data, n_regimes_max=10):
#     """
#     Build a Multivariate GARCH model with Infinite Hidden Markov Model

#     Parameters:
#     data (pd.DataFrame): Preprocessed financial data
#     n_regimes_max (int): Maximum number of regimes to consider

#     Returns:
#     pm.Model: PyMC model object
#     """
#     n_features = data.shape[1]
#     n_observations = len(data)

#     with pm.Model() as model:
#         # Dirichlet Process concentration parameter (controls how many regimes we expect)
#         alpha = pm.Gamma("alpha", alpha=2.0, beta=1.0)

#         # Stick-breaking process for IHMM
#         beta = pm.Beta("beta", alpha=1.0, beta=alpha, shape=n_regimes_max - 1)
#         w = pm.Deterministic("w", beta * pt.concatenate([[1], pt.cumprod(1 - beta)]))

#         # Regime-specific parameters
#         mu = pm.Normal("mu", mu=0, sigma=1, shape=(n_regimes_max, n_features))

#         # GARCH parameters for each regime
#         omega = pm.HalfNormal("omega", sigma=0.2, shape=(n_regimes_max, n_features))
#         alpha_garch = pm.HalfNormal(
#             "alpha_garch", sigma=0.2, shape=(n_regimes_max, n_features)
#         )
#         beta_garch = pm.HalfNormal(
#             "beta_garch", sigma=0.5, shape=(n_regimes_max, n_features)
#         )

#         # Correlation matrices for each regime (using LKJ distribution)
#         chol_packed = []
#         correlation = []
#         for k in range(n_regimes_max):
#             chol, corr, chol_stds = pm.LKJCholeskyCov(
#                 f"chol_packed_{k}",
#                 n=n_features,
#                 eta=2.0,
#                 sd_dist=pm.HalfCauchy.dist(1.0),
#             )
#             chol_packed.append(chol)

#             L = pm.expand_packed_triangular(n_features, chol_packed[k].ravel())
#             corr_matrix = L @ pt.transpose(L)
#             correlation.append(pm.Deterministic(f"correlation_{k}", corr_matrix))

#         # Initial state probabilities (stationary distribution)
#         initial_probs = pm.Dirichlet("initial_probs", a=np.ones(n_regimes_max))

#         # Transition matrix with sticky property
#         kappa = pm.Gamma("kappa", alpha=50, beta=10)  # Stickiness parameter

#         # Create transition matrix with diagonal stickiness
#         remaining_stick = 1 - pt.sum(w)
#         w_full = pt.concatenate([w, [remaining_stick]])
#         transition_base = pt.tile(w_full, (n_regimes_max, 1))

#         diag_mask = pt.eye(n_regimes_max)
#         transition_probs = pm.Deterministic(
#             "transition_probs",
#             transition_base * (1 - kappa * diag_mask) + diag_mask * kappa,
#         )

#         # Manual implementation of Markov states
#         # First state from initial distribution
#         states = [pm.Categorical("state_0", p=initial_probs)]

#         # Rest of states follow Markov property
#         for t in range(1, n_observations):
#             # Get transition probabilities from previous state
#             trans_p = transition_probs[states[-1]]

#             # Sample current state
#             state_t = pm.Categorical(f"state_{t}", p=trans_p)
#             states.append(state_t)

#         # Since we can't directly combine the list of RVs into a tensor within PyMC,
#         # we'll access them individually in the sampling phase.
#         # But we can add a hack for calculating volatility...

#         # Initialize values for GARCH volatility calculation
#         data_tensor = pt.as_tensor_variable(data.values)
#         h = pt.zeros((n_observations, n_features))

#         # Initialize with empirical variance (or some reasonable value)
#         h = pt.set_subtensor(h[0], pt.mean(data_tensor**2, axis=0))

#         # Calculate volatility for each time step (this is a simplified version)
#         for t in range(1, n_observations):
#             regime = states[t - 1]
#             h = pt.set_subtensor(
#                 h[t],
#                 omega[regime]
#                 + alpha_garch[regime] * pt.sqr(data_tensor[t - 1])
#                 + beta_garch[regime] * h[t - 1],
#             )

#         # Store volatility as deterministic
#         volatility = pm.Deterministic("volatility", h)

#     return model

# def build_mgarch_ihmm_model(data, n_regimes_max=10):
#     """
#     Build a Multivariate GARCH model with Infinite Hidden Markov Model

#     Parameters:
#     data (pd.DataFrame): Preprocessed financial data
#     n_regimes_max (int): Maximum number of regimes to consider

#     Returns:
#     pm.Model: PyMC model object
#     """
#     n_features = data.shape[1]
#     n_observations = len(data)

#     with pm.Model() as model:
#         # Dirichlet Process concentration parameter
#         alpha = pm.Gamma("alpha", alpha=2.0, beta=1.0)

#         # Stick-breaking process for IHMM - ensure we have exactly n_regimes_max elements
#         beta = pm.Beta("beta", alpha=1.0, beta=alpha, shape=n_regimes_max - 1)
#         w_partial = beta * pt.concatenate([[1], pt.cumprod(1 - beta)[:-1]])
#         # Make sure remaining probability is included to sum to 1
#         remaining_prob = 1 - pt.sum(w_partial)
#         w = pm.Deterministic("w", pt.concatenate([w_partial, [remaining_prob]]))

#         # Regime-specific parameters
#         mu = pm.Normal("mu", mu=0, sigma=1, shape=(n_regimes_max, n_features))

#         # GARCH parameters
#         omega = pm.HalfNormal("omega", sigma=0.2, shape=(n_regimes_max, n_features))
#         alpha_garch = pm.HalfNormal(
#             "alpha_garch", sigma=0.2, shape=(n_regimes_max, n_features)
#         )
#         beta_garch = pm.HalfNormal(
#             "beta_garch", sigma=0.5, shape=(n_regimes_max, n_features)
#         )

#         # Correlation matrices
#         chol_packed = []
#         correlation = []
#         for k in range(n_regimes_max):
#             chol, corr, chol_stds = pm.LKJCholeskyCov(
#                 f"chol_packed_{k}",
#                 n=n_features,
#                 eta=2.0,
#                 sd_dist=pm.HalfCauchy.dist(1.0),
#             )
#             chol_packed.append(chol)
#             L = pm.expand_packed_triangular(n_features, chol_packed[k].ravel())
#             corr_matrix = L @ pt.transpose(L)
#             correlation.append(pm.Deterministic(f"correlation_{k}", corr_matrix))

#         # Initial state probabilities
#         initial_probs = pm.Dirichlet("initial_probs", a=np.ones(n_regimes_max))

#         # Transition matrix with sticky property
#         kappa = pm.Gamma("kappa", alpha=50, beta=10)  # Stickiness parameter

#         # Create transition matrix - ensure proper dimensions
#         transition_base = pt.tile(w, (n_regimes_max, 1))
#         diag_mask = pt.eye(n_regimes_max)
#         transition_probs = pm.Deterministic(
#             "transition_probs",
#             transition_base * (1 - kappa * diag_mask) + diag_mask * kappa,
#         )

#         # First state
#         states = [pm.Categorical("state_0", p=initial_probs)]

#         # Create subsequent states
#         for t in range(1, n_observations):
#             # Make sure to use proper indexing
#             prev_state = states[-1]
#             # Add bounds checking to prevent out-of-bounds errors
#             prev_state_safe = pt.clip(prev_state, 0, n_regimes_max - 1)
#             # Get transition probabilities
#             trans_p = transition_probs[prev_state_safe]
#             # Sample next state
#             next_state = pm.Categorical(f"state_{t}", p=trans_p)
#             states.append(next_state)

#     return model


def build_mgarch_ihmm_model(data, n_regimes_max=10):
    """
    Build a Multivariate GARCH model with Infinite Hidden Markov Model
    Parameters:
    data (pd.DataFrame): Preprocessed financial data
    n_regimes_max (int): Maximum number of regimes to consider
    Returns:
    pm.Model: PyMC model object
    """
    n_features = data.shape[1]
    n_observations = len(data)
    with pm.Model() as model:
        # Dirichlet Process concentration parameter
        alpha = pm.Gamma("alpha", alpha=2.0, beta=1.0)

        # Stick-breaking process for IHMM
        beta = pm.Beta("beta", alpha=1.0, beta=alpha, shape=n_regimes_max - 1)

        # Manual stick-breaking implementation
        # First weight is just beta[0]
        w = pt.zeros(n_regimes_max)
        w = pt.set_subtensor(w[0], beta[0])

        # Calculate remaining weights
        cumprod_1m_beta = pt.cumprod(1 - beta)
        for i in range(1, n_regimes_max - 1):
            w = pt.set_subtensor(w[i], beta[i] * cumprod_1m_beta[i - 1])

        # Last weight (ensures sum to 1)
        w = pt.set_subtensor(w[n_regimes_max - 1], cumprod_1m_beta[n_regimes_max - 2])

        # Store as deterministic
        w = pm.Deterministic("w", w)

        # Regime-specific parameters
        mu = pm.Normal("mu", mu=0, sigma=1, shape=(n_regimes_max, n_features))

        # GARCH parameters
        omega = pm.HalfNormal("omega", sigma=0.2, shape=(n_regimes_max, n_features))
        alpha_garch = pm.HalfNormal(
            "alpha_garch", sigma=0.2, shape=(n_regimes_max, n_features)
        )
        beta_garch = pm.HalfNormal(
            "beta_garch", sigma=0.5, shape=(n_regimes_max, n_features)
        )

        # Correlation matrices
        chol_packed = []
        correlation = []
        for k in range(n_regimes_max):
            chol, corr, chol_stds = pm.LKJCholeskyCov(
                f"chol_packed_{k}",
                n=n_features,
                eta=2.0,
                sd_dist=pm.HalfCauchy.dist(1.0),
            )
            chol_packed.append(chol)
            L = pm.expand_packed_triangular(n_features, chol_packed[k].ravel())
            corr_matrix = L @ pt.transpose(L)
            correlation.append(pm.Deterministic(f"correlation_{k}", corr_matrix))

        # Initial state probabilities
        initial_probs = pm.Dirichlet("initial_probs", a=np.ones(n_regimes_max))

        # Transition matrix with sticky property
        kappa = pm.Gamma("kappa", alpha=50, beta=10)  # Stickiness parameter

        # Create transition matrix - ensure proper dimensions
        transition_base = pt.tile(w, (n_regimes_max, 1))
        diag_mask = pt.eye(n_regimes_max)
        transition_probs = pm.Deterministic(
            "transition_probs",
            transition_base * (1 - kappa * diag_mask) + diag_mask * kappa,
        )

        # First state
        states = [pm.Categorical("state_0", p=initial_probs)]

        # Create subsequent states
        for t in range(1, n_observations):
            # Make sure to use proper indexing
            prev_state = states[-1]
            # Add bounds checking to prevent out-of-bounds errors
            prev_state_safe = pt.clip(prev_state, 0, n_regimes_max - 1)
            # Get transition probabilities
            trans_p = transition_probs[prev_state_safe]
            # Sample next state
            next_state = pm.Categorical(f"state_{t}", p=trans_p)
            states.append(next_state)

    return model


def update_garch_params(trace, data, states):
    """
    Update GARCH parameters based on posterior samples

    Parameters:
    trace: PyMC trace object
    data (pd.DataFrame): Data
    states (np.ndarray): Regime assignments

    Returns:
    np.ndarray: Updated volatility matrix
    """
    # Extract GARCH parameters
    omega_samples = (
        az.extract(trace, var_names=["omega"]).mean(dim=["chain", "draw"]).values
    )
    alpha_samples = (
        az.extract(trace, var_names=["alpha_garch"]).mean(dim=["chain", "draw"]).values
    )
    beta_samples = (
        az.extract(trace, var_names=["beta_garch"]).mean(dim=["chain", "draw"]).values
    )

    # Convert to PyTorch tensors
    data_tensor = torch.tensor(data.values, dtype=torch.float32)
    states_tensor = torch.tensor(states, dtype=torch.long)
    omega_tensor = torch.tensor(omega_samples, dtype=torch.float32)
    alpha_tensor = torch.tensor(alpha_samples, dtype=torch.float32)
    beta_tensor = torch.tensor(beta_samples, dtype=torch.float32)

    # Calculate GARCH volatilities
    n_obs = data_tensor.shape[0]
    n_feat = data_tensor.shape[1]
    h = torch.zeros((n_obs, n_feat))

    # Initialize with empirical variance
    h[0] = torch.mean(data_tensor**2, dim=0)

    for t in range(1, n_obs):
        regime = states_tensor[t - 1]
        h[t] = (
            omega_tensor[regime]
            + alpha_tensor[regime] * data_tensor[t - 1] ** 2
            + beta_tensor[regime] * h[t - 1]
        )

    return h.numpy()


def sample_mgarch_ihmm(model, tune=1000, draws=1000, cores=4):
    """
    Sample from the MGARCH-IHMM model with enhanced error handling and logging.

    Parameters:
    model (pm.Model): PyMC model object
    tune (int): Number of tuning samples
    draws (int): Number of draws
    cores (int): Number of CPU cores to use

    Returns:
    InferenceData: MCMC samples
    """
    with model:

        # Debugging print statements
        print("\nStarting model compilation...\n")

        # Separate continuous and categorical variables
        continuous_vars = []
        categorical_vars = []

        for var in model.free_RVs:
            if var.dtype in ("int64", "int32", "int16", "int8"):
                categorical_vars.append(var)
            else:
                try:
                    continuous_vars.append(var)
                except Exception as e:
                    print(f"Error while categorizing variable {var.name}: {e}")
                    raise

        # Define step methods explicitly
        steps = []
        if len(continuous_vars) > 0:
            steps.append(pm.NUTS(vars=continuous_vars, target_accept=0.8))

        if len(categorical_vars) > 0:
            try:
                steps.append(pm.CategoricalGibbsMetropolis(vars=categorical_vars))
            except Exception as e:
                print(f"Error in creating CategoricalGibbsMetropolis step: {e}")
                raise

        # Compile the function for updates
        # f = pt.shared_function(model, tune, draws)

        # if cores > 1:
        #     # Check if PyMC supports multiprocessing with NUTS and Gibbs
        #     try:
        #         pm.setmp(cores=cores)
        #         print(f"\nInitializing sampling with {cores} CPU cores...\n")
        #     except Exception as e:
        #         print(f"Error setting up multiple cores: {e}")
        #         raise

        # Sample with explicit step methods
        start_time = time()
        try:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=min(4, cores),
                cores=cores,
                step=steps,
                return_inferencedata=True,
            )
        except Exception as e:
            print(f"Sampling failed with error: {e}")
            raise
        print(f"\nCompleted in {(time() - start_time):.2f} seconds\n")

        # Debugging: Print the number of draws and tuning samples
        print(f"Total samples: {len(idata)}")
        for var in idata:
            print(f"InferenceData has {var}")

    return idata


# def sample_mgarch_ihmm(model, tune=1000, draws=1000, cores=4):
#     """
#     Sample from the MGARCH-IHMM model

#     Parameters:
#     model (pm.Model): PyMC model object
#     tune (int): Number of tuning samples
#     draws (int): Number of draws
#     cores (int): Number of CPU cores to use

#     Returns:
#     InferenceData: MCMC samples
#     """
#     with model:
#         # Separate continuous and categorical variables
#         continuous_vars = []
#         categorical_vars = []

#         for var in model.free_RVs:
#             if var.dtype in ("int64", "int32", "int16", "int8"):
#                 categorical_vars.append(var)
#             else:
#                 continuous_vars.append(var)

#         # Define step methods explicitly
#         steps = []
#         if continuous_vars:
#             steps.append(pm.NUTS(vars=continuous_vars, target_accept=0.8))
#         if categorical_vars:
#             steps.append(pm.CategoricalGibbsMetropolis(vars=categorical_vars))

#         # Sample with explicit step methods
#         idata = pm.sample(
#             draws=draws,
#             tune=tune,
#             chains=min(4, cores),
#             cores=cores,
#             step=steps,
#             return_inferencedata=True,
#         )

#     return idata


# def sample_mgarch_ihmm(model, tune=1000, draws=1000, cores=4):
#     """
#     Sample from the MGARCH-IHMM model using PyMC 4.

#     Parameters:
#     model (pm.Model): PyMC model object
#     tune (int): Number of tuning samples
#     draws (int): Number of draws
#     cores (int): Number of CPU cores to use

#     Returns:
#     InferenceData: MCMC samples
#     """
#     with model:
#         # Use NUTS for all variables
#         idata = pm.sample(
#             draws=draws,
#             tune=tune,
#             chains=min(4, cores),
#             cores=cores,
#             step=pm.NUTS(vars=model.free_RVs),
#             return_inferencedata=True,
#         )

#     return idata


def analyze_regimes(trace, data):
    """
    Analyze identified regimes from the MGARCH-IHMM model

    Parameters:
    trace: PyMC trace object
    data (pd.DataFrame): Original data

    Returns:
    pd.DataFrame: Data with regime assignments
    """
    # Extract state sequence
    states_posterior = az.extract(trace, var_names=["states"])
    states_mean = states_posterior.mean(dim=["chain", "draw"]).values

    # Get most probable regime for each time point
    regime_assignments = np.round(states_mean).astype(int)

    # Update GARCH volatilities with extracted states
    h = update_garch_params(trace, data, regime_assignments)

    # Add regime assignments to data
    data_with_regimes = data.copy()
    data_with_regimes["regime"] = regime_assignments

    # Add volatility estimates
    for i in range(data.shape[1]):
        data_with_regimes[f"vol_{data.columns[i]}"] = np.sqrt(h[:, i])

    # Count occurrences of each regime
    unique_regimes = np.unique(regime_assignments)
    regime_counts = {
        regime: np.sum(regime_assignments == regime) for regime in unique_regimes
    }

    print(f"Identified {len(unique_regimes)} distinct regimes")
    for regime, count in regime_counts.items():
        print(f"Regime {regime}: {count} days ({count/len(data)*100:.2f}%)")

    return data_with_regimes


def plot_regime_transitions(data_with_regimes, returns_col="returns"):
    """
    Plot regime transitions and characteristics

    Parameters:
    data_with_regimes (pd.DataFrame): Data with regime assignments
    returns_col (str): Column name for returns
    """
    # Plot returns with regime colors
    regimes = sorted(data_with_regimes["regime"].unique())
    cmap = plt.cm.get_cmap("viridis", len(regimes))

    plt.figure(figsize=(15, 10))

    # Plot 1: Returns colored by regime
    plt.subplot(2, 1, 1)
    for regime in regimes:
        regime_data = data_with_regimes[data_with_regimes["regime"] == regime]
        plt.scatter(
            regime_data.index,
            regime_data[returns_col],
            color=cmap(regime / len(regimes)),
            alpha=0.7,
            s=3,
            label=f"Regime {regime}",
        )

    plt.title("Returns by Regime")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Regime transitions over time
    plt.subplot(2, 1, 2)
    plt.plot(
        data_with_regimes.index, data_with_regimes["regime"], drawstyle="steps-post"
    )
    plt.yticks(regimes)
    plt.title("Regime Transitions Over Time")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calculate regime transition probabilities
    regime_shifts = np.diff(data_with_regimes["regime"].values)
    transitions = np.zeros((len(regimes), len(regimes)))

    for i in range(len(data_with_regimes) - 1):
        from_regime = data_with_regimes["regime"].iloc[i]
        to_regime = data_with_regimes["regime"].iloc[i + 1]
        transitions[from_regime, to_regime] += 1

    # Normalize to get probabilities
    row_sums = transitions.sum(axis=1, keepdims=True)
    transition_probs = np.zeros_like(transitions)
    np.divide(transitions, row_sums, out=transition_probs, where=row_sums != 0)

    # Plot transition probability matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        transition_probs,
        annot=True,
        cmap="coolwarm",
        xticklabels=[f"Regime {r}" for r in regimes],
        yticklabels=[f"Regime {r}" for r in regimes],
    )
    plt.title("Regime Transition Probabilities")
    plt.xlabel("To Regime")
    plt.ylabel("From Regime")
    plt.tight_layout()
    plt.show()

    # Plot volatility by regime
    vol_cols = [col for col in data_with_regimes.columns if col.startswith("vol_")]

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(vol_cols):
        plt.subplot(2, 3, i + 1)
        for regime in regimes:
            regime_data = data_with_regimes[data_with_regimes["regime"] == regime]
            sns.kdeplot(regime_data[col], label=f"Regime {regime}")
        plt.title(f"Distribution of {col} by Regime")
        plt.legend()

    plt.tight_layout()
    plt.show()

    # Analyze regime characteristics
    feature_cols = [
        col
        for col in data_with_regimes.columns
        if col not in vol_cols and col != "regime"
    ]

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_cols):
        plt.subplot(2, 3, i + 1)
        for regime in regimes:
            regime_data = data_with_regimes[data_with_regimes["regime"] == regime]
            sns.kdeplot(regime_data[col], label=f"Regime {regime}")
        plt.title(f"Distribution of {col} by Regime")
        plt.legend()

    plt.tight_layout()
    plt.show()


# Main execution function
def run_mgarch_ihmm_analysis(df, n_regimes_max=10, tune=1000, draws=1000, cores=4):
    """
    Run the complete MGARCH-IHMM analysis pipeline

    Parameters:
    df (pd.DataFrame): OHLCV data with RSI, MACD, and Hurst exponent
    n_regimes_max (int): Maximum number of regimes to consider
    tune (int): Number of tuning samples
    draws (int): Number of draws
    cores (int): Number of CPU cores to use

    Returns:
    pd.DataFrame: Data with regime assignments
    """
    # Step 1: Preprocess data
    print("Preprocessing data...")
    processed_data, scaler = preprocess_financial_data(df)

    # Step 2: Build model
    print("Building MGARCH-IHMM model...")
    model = build_mgarch_ihmm_model(processed_data, n_regimes_max=n_regimes_max)

    # Step 3: Sample from posterior
    print(f"Sampling from posterior (this may take a while)...")
    trace = sample_mgarch_ihmm(model, tune=tune, draws=draws, cores=cores)

    # Step 4: Analyze regimes
    print("Analyzing identified regimes...")
    data_with_regimes = analyze_regimes(trace, processed_data)

    # Step 5: Visualize results
    print("Generating visualizations...")
    plot_regime_transitions(data_with_regimes)

    # Inverse transform to original scale if needed
    feature_cols = [
        col
        for col in data_with_regimes.columns
        if not col.startswith("vol_") and col != "regime"
    ]

    data_with_regimes_original = pd.DataFrame(
        scaler.inverse_transform(data_with_regimes[feature_cols]),
        index=data_with_regimes.index,
        columns=feature_cols,
    )

    # Keep regime and volatility columns
    data_with_regimes_original["regime"] = data_with_regimes["regime"]
    for vol_col in [col for col in data_with_regimes.columns if col.startswith("vol_")]:
        data_with_regimes_original[vol_col] = data_with_regimes[vol_col]

    return data_with_regimes_original, trace


def umap_mgarch_ihmm_analysis(df, n_regimes_max=10, tune=1000, draws=1000, cores=4):
    """
    Run the complete MGARCH-IHMM analysis pipeline

    Parameters:
    df (pd.DataFrame): OHLCV data with RSI, MACD, and Hurst exponent
    n_regimes_max (int): Maximum number of regimes to consider
    tune (int): Number of tuning samples
    draws (int): Number of draws
    cores (int): Number of CPU cores to use

    Returns:
    pd.DataFrame: Data with regime assignments
    """
    # Step 1: Preprocess data

    # Step 2: Build model
    print("Building MGARCH-IHMM model...")
    model = build_mgarch_ihmm_model(df, n_regimes_max=n_regimes_max)

    # Step 3: Sample from posterior
    print(f"Sampling from posterior (this may take a while)...")
    trace = sample_mgarch_ihmm(model, tune=tune, draws=draws, cores=cores)

    # Step 4: Analyze regimes
    print("Analyzing identified regimes...")
    data_with_regimes = analyze_regimes(trace, df)

    # Step 5: Visualize results
    print("Generating visualizations...")
    plot_regime_transitions(data_with_regimes)

    return data_with_regimes, trace


# Function to evaluate regime performance
def evaluate_regime_performance(data_with_regimes, returns_col="returns"):
    """
    Evaluate trading performance by regime

    Parameters:
    data_with_regimes (pd.DataFrame): Data with regime assignments
    returns_col (str): Column name for returns

    Returns:
    pd.DataFrame: Performance metrics by regime
    """
    regimes = sorted(data_with_regimes["regime"].unique())

    # Calculate performance metrics by regime
    metrics = []
    for regime in regimes:
        regime_data = data_with_regimes[data_with_regimes["regime"] == regime]

        # Skip if not enough data
        if len(regime_data) < 5:
            continue

        annualized_factor = 252  # Assuming daily data

        # Calculate metrics
        mean_return = regime_data[returns_col].mean()
        std_return = regime_data[returns_col].std()
        annualized_return = mean_return * annualized_factor
        annualized_vol = std_return * np.sqrt(annualized_factor)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Calculate drawdowns
        cum_returns = (1 + regime_data[returns_col]).cumprod()
        running_max = cum_returns.cummax()
        drawdowns = cum_returns / running_max - 1
        max_drawdown = drawdowns.min()

        # Calculate win rate
        win_rate = (regime_data[returns_col] > 0).mean()

        metrics.append(
            {
                "Regime": regime,
                "Count": len(regime_data),
                "Mean Return": mean_return,
                "Std Dev": std_return,
                "Annualized Return": annualized_return,
                "Annualized Vol": annualized_vol,
                "Sharpe Ratio": sharpe,
                "Max Drawdown": max_drawdown,
                "Win Rate": win_rate,
            }
        )

    return pd.DataFrame(metrics).set_index("Regime")


# Example usage
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# data_with_regimes, trace = run_mgarch_ihmm_analysis(df)
# performance_metrics = evaluate_regime_performance(data_with_regimes)
# print(performance_metrics)


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
        hurst_x_scaled, hurst_values, kind="slinear", fill_value="extrapolate"
    )

    # Apply interpolation to get Hurst values aligned with price data
    aligned_hurst = interp_func(prices_x)

    return aligned_hurst


def prepare_data(data):
    """Prepares the data for TDA by selecting and scaling features."""
    # Drop rows with missing values

    hurst = rolling_hurst(
        data["Close"], window_size=63, step=5, min_rs_window=5, max_rs_window=10
    )
    print(hurst.head())
    data["ema_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["ema_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["ema_12"] - data["ema_26"]
    data["Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["Hist"] = data["MACD"] - data["Signal"]
    delta = data["Close"].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    avg_gain = up.rolling(window=14).mean()
    avg_loss = abs(down.rolling(window=14).mean())
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))
    data["returns"] = data["Close"].pct_change()
    hurst = align_hurst_to_prices(data["Close"], hurst["Hurst"])
    data["Hurst"] = hurst
    data = data.dropna()
    data.columns = data.columns.str.lower()
    return data


# df = pd.read_csv(
#     "nasdaq_futures_1D_Close_20250226.csv", index_col="Date", parse_dates=True
# )
# df = prepare_data(df)
file = "market_regime_det.pkl"

with open(file, "rb") as f:
    res = pickle.load(f)

df = res["dimensionality_reduction"]["umap"]["n15_d0.5"]
data_with_regimes, trace = umap_mgarch_ihmm_analysis(df)
