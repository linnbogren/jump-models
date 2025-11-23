"""
Simulation Utilities for Sparse Poisson Jump Model Evaluation

This module provides core utilities for generating synthetic data and computing
evaluation metrics for the simulation study described in simulation_instructions.tex.

Key components:
- HMM transition matrix generation
- Poisson HMM data generation with state-specific rates
- Overdispersed (Negative Binomial) data generation
- Correlated noise generation using Gaussian Copula
- Evaluation metrics (BAC, persistence, feature selection, deviance, stability)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import factorial
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, asdict
from itertools import permutations


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""
    n_samples: int = 600
    n_states: int = 3
    n_informative: int = 15
    n_noise: int = 0  # Derived from n_total_features
    n_total_features: int = 15
    delta: float = 0.5
    lambda_0: float = 10.0
    persistence: float = 0.97
    distribution_type: str = "Poisson"  # "Poisson", "NegativeBinomial", "Gaussian"
    correlated_noise: bool = False
    noise_correlation: float = 0.1
    nb_dispersion: float = 2.0  # For Negative Binomial
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        """Calculate derived parameters."""
        self.n_noise = self.n_total_features - self.n_informative
        assert self.n_noise >= 0, "n_total_features must be >= n_informative"
        assert 0 <= self.delta < 1, "delta must be in [0, 1)"
        assert self.n_states >= 2, "Must have at least 2 states"


@dataclass
class ReplicationResult:
    """Results from a single replication (best model only)."""
    config: SimulationConfig
    model_name: str
    balanced_accuracy: float
    accuracy: float
    n_jumps_true: int
    n_jumps_estimated: int
    persistence_error: int
    feature_f1: float
    feature_precision: float
    feature_recall: float
    n_selected_noise: int
    poisson_deviance: float
    computation_time: float
    selected_features: List[int]
    true_states: np.ndarray
    predicted_states: np.ndarray
    best_hyperparams: Dict  # Hyperparameters of the best model
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        d = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        d['true_states'] = self.true_states.tolist() if hasattr(self.true_states, 'tolist') else self.true_states
        d['predicted_states'] = self.predicted_states.tolist() if hasattr(self.predicted_states, 'tolist') else self.predicted_states


@dataclass
class GridSearchResult:
    """Results from all models in a grid search."""
    config: SimulationConfig
    model_name: str
    hyperparameters: Dict  # The hyperparameters for this model
    balanced_accuracy: float
    accuracy: float
    n_jumps_true: int
    n_jumps_estimated: int
    persistence_error: int
    n_breakpoints_true: int  # Number of true breakpoints
    n_breakpoints_estimated: int  # Number of estimated breakpoints
    breakpoint_count_error: int  # Absolute difference in breakpoint counts
    chamfer_distance: float  # Chamfer distance between true and estimated breakpoints
    feature_f1: float
    feature_precision: float
    feature_recall: float
    n_selected_noise: int
    n_selected_total: int
    computation_time: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return asdict(self)
        return d


###############################################################################
# Transition Matrix Generation
###############################################################################

def generate_hmm_transition_matrix(n_states: int, 
                                   persistence: float = 0.97) -> np.ndarray:
    """
    Generate a transition matrix with high diagonal dominance.
    
    The matrix has diagonal elements approximately equal to `persistence`,
    with off-diagonal elements uniformly distributed to sum to (1 - persistence).
    
    Parameters:
    -----------
    n_states : int
        Number of states in the HMM.
    persistence : float
        Probability of staying in the same state (diagonal elements).
        
    Returns:
    --------
    np.ndarray
        Transition matrix of shape (n_states, n_states) where rows sum to 1.
    """
    assert 0 < persistence < 1, "Persistence must be in (0, 1)"
    assert n_states >= 2, "Must have at least 2 states"
    
    A = np.zeros((n_states, n_states))
    
    # Set diagonal to persistence
    np.fill_diagonal(A, persistence)
    
    # Distribute remaining probability uniformly to off-diagonal elements
    off_diag_prob = (1 - persistence) / (n_states - 1)
    A = A + off_diag_prob * (1 - np.eye(n_states))
    
    # Ensure rows sum to 1 (handle floating point errors)
    A = A / A.sum(axis=1, keepdims=True)
    
    return A


###############################################################################
# Lambda (Rate Parameter) Generation
###############################################################################

def compute_state_lambdas(lambda_0: float,
                          delta: float,
                          n_states: int,
                          n_features: int) -> np.ndarray:
    """
    Compute state-specific Poisson rates for the informative features.
    
    Following the pattern in simulation_instructions.tex:
    - State 1: λ₀(1 - δ) - low rate
    - State 2: λ₀ - baseline rate (overlaps with noise)
    - State 3: λ₀(1 + δ) - high rate
    - For more states, interpolate linearly
    
    Parameters:
    -----------
    lambda_0 : float
        Baseline Poisson rate.
    delta : float
        Signal strength parameter (0 <= delta < 1).
    n_states : int
        Number of states.
    n_features : int
        Number of informative features.
        
    Returns:
    --------
    np.ndarray
        Array of shape (n_states, n_features) with state-specific rates.
    """
    assert 0 <= delta < 1, "delta must be in [0, 1)"
    assert lambda_0 > 0, "lambda_0 must be positive"
    
    lambdas = np.zeros((n_states, n_features))
    
    if n_states == 2:
        # Two states: low and high
        lambdas[0, :] = lambda_0 * (1 - delta)
        lambdas[1, :] = lambda_0 * (1 + delta)
    elif n_states == 3:
        # Three states: low, baseline, high
        lambdas[0, :] = lambda_0 * (1 - delta)
        lambdas[1, :] = lambda_0
        lambdas[2, :] = lambda_0 * (1 + delta)
    else:
        # More states: interpolate linearly from (1-δ) to (1+δ)
        multipliers = np.linspace(1 - delta, 1 + delta, n_states)
        for k in range(n_states):
            lambdas[k, :] = lambda_0 * multipliers[k]
    
    return lambdas


###############################################################################
# Data Generation
###############################################################################

def sample_state_sequence(n_samples: int,
                          n_states: int,
                          transition_matrix: np.ndarray,
                          random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample a state sequence from an HMM with given transition matrix.
    
    Parameters:
    -----------
    n_samples : int
        Length of the sequence.
    n_states : int
        Number of states.
    transition_matrix : np.ndarray
        Transition matrix of shape (n_states, n_states).
    random_state : int, optional
        Random seed.
        
    Returns:
    --------
    states : np.ndarray
        State sequence of shape (n_samples,).
    breakpoints : np.ndarray
        Indices where state changes occur.
    """
    rng = np.random.RandomState(random_state)
    
    states = np.zeros(n_samples, dtype=int)
    breakpoints = []
    
    # Start with uniform distribution over states
    states[0] = rng.choice(n_states)
    
    # Generate sequence using transition matrix
    for t in range(1, n_samples):
        states[t] = rng.choice(n_states, p=transition_matrix[states[t-1], :])
        if states[t] != states[t-1]:
            breakpoints.append(t)
    
    return states, np.array(breakpoints, dtype=int)


def generate_poisson_hmm_data(config: SimulationConfig) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate data from a Poisson HMM.
    
    Parameters:
    -----------
    config : SimulationConfig
        Configuration object with all parameters.
        
    Returns:
    --------
    X : pd.DataFrame
        Data matrix of shape (n_samples, n_total_features).
    states : np.ndarray
        True state sequence.
    breakpoints : np.ndarray
        True breakpoint indices.
    """
    rng = np.random.RandomState(config.random_seed)
    
    # Generate transition matrix
    A = generate_hmm_transition_matrix(config.n_states, config.persistence)
    
    # Sample state sequence
    states, breakpoints = sample_state_sequence(
        config.n_samples, config.n_states, A, config.random_seed
    )
    
    # Compute state-specific lambdas for informative features
    lambdas_inform = compute_state_lambdas(
        config.lambda_0, config.delta, config.n_states, config.n_informative
    )
    
    # Initialize data matrix
    X = np.zeros((config.n_samples, config.n_total_features))
    
    # Generate informative features
    for t in range(config.n_samples):
        state_t = states[t]
        X[t, :config.n_informative] = rng.poisson(lambdas_inform[state_t, :])
    
    # Generate noise features (constant rate λ₀ across all states)
    if config.n_noise > 0:
        if config.correlated_noise:
            X[:, config.n_informative:] = generate_correlated_noise(
                config.n_samples, config.n_noise, config.lambda_0,
                config.noise_correlation, rng
            )
        else:
            X[:, config.n_informative:] = rng.poisson(
                config.lambda_0, size=(config.n_samples, config.n_noise)
            )
    
    # Create DataFrame with proper column names
    col_names = ([f'informative_{i+1}' for i in range(config.n_informative)] +
                 [f'noise_{i+1}' for i in range(config.n_noise)])
    X_df = pd.DataFrame(X, columns=col_names)
    
    return X_df, states, breakpoints


def generate_negative_binomial_hmm_data(config: SimulationConfig) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate overdispersed count data using Negative Binomial distribution.
    
    The Negative Binomial is parameterized to match the Poisson mean but with
    increased variance (overdispersion).
    
    Parameters:
    -----------
    config : SimulationConfig
        Configuration object. Uses nb_dispersion parameter for variance inflation.
        
    Returns:
    --------
    X : pd.DataFrame
        Data matrix with overdispersed counts.
    states : np.ndarray
        True state sequence.
    breakpoints : np.ndarray
        True breakpoint indices.
    """
    rng = np.random.RandomState(config.random_seed)
    
    # Generate transition matrix and states
    A = generate_hmm_transition_matrix(config.n_states, config.persistence)
    states, breakpoints = sample_state_sequence(
        config.n_samples, config.n_states, A, config.random_seed
    )
    
    # Compute lambdas
    lambdas_inform = compute_state_lambdas(
        config.lambda_0, config.delta, config.n_states, config.n_informative
    )
    
    X = np.zeros((config.n_samples, config.n_total_features))
    
    # For Negative Binomial: Var = μ + μ²/r
    # We want Var = φ * μ, so r = μ / (φ - 1)
    phi = config.nb_dispersion
    
    # Generate informative features
    for t in range(config.n_samples):
        state_t = states[t]
        for f in range(config.n_informative):
            mu = lambdas_inform[state_t, f]
            r = mu / (phi - 1) if phi > 1 else 1e6  # Large r ≈ Poisson
            p = r / (r + mu)
            X[t, f] = rng.negative_binomial(r, p)
    
    # Generate noise features
    if config.n_noise > 0:
        mu_noise = config.lambda_0
        r_noise = mu_noise / (phi - 1) if phi > 1 else 1e6
        p_noise = r_noise / (r_noise + mu_noise)
        X[:, config.n_informative:] = rng.negative_binomial(
            r_noise, p_noise, size=(config.n_samples, config.n_noise)
        )
    
    col_names = ([f'informative_{i+1}' for i in range(config.n_informative)] +
                 [f'noise_{i+1}' for i in range(config.n_noise)])
    X_df = pd.DataFrame(X, columns=col_names)
    
    return X_df, states, breakpoints


def generate_gaussian_hmm_data(config: SimulationConfig) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Generate data from a Gaussian HMM.
    
    Similar to the example in the paper attachment:
    y_t | s_t ~ N(μ_{s_t}, I_P)
    
    where μ_1 = μ·1_{p≤15}, μ_2 = 0, μ_3 = -μ·1_{p≤15}
    
    Parameters:
    -----------
    config : SimulationConfig
        Configuration object with all parameters.
        Uses lambda_0 as the mean μ for state separation.
        Uses delta to control separation: μ = lambda_0 * (1 + delta) / (1 - delta)
        
    Returns:
    --------
    X : pd.DataFrame
        Data matrix of shape (n_samples, n_total_features).
    states : np.ndarray
        True state sequence.
    breakpoints : np.ndarray
        True breakpoint indices.
    """
    rng = np.random.RandomState(config.random_seed)
    
    # Generate transition matrix
    A = generate_hmm_transition_matrix(config.n_states, config.persistence)
    
    # Sample state sequence
    states, breakpoints = sample_state_sequence(
        config.n_samples, config.n_states, A, config.random_seed
    )
    
    # Compute state-specific means for informative features
    # For Gaussian, we adapt the Poisson lambda computation
    # to create separated means: μ_k = λ_k for k=1,2,3
    means_inform = compute_state_lambdas(
        config.lambda_0, config.delta, config.n_states, config.n_informative
    )
    
    # Initialize data matrix
    X = np.zeros((config.n_samples, config.n_total_features))
    
    # Generate informative features: y_t ~ N(μ_{s_t}, I)
    for t in range(config.n_samples):
        state_t = states[t]
        # Sample from N(μ_{s_t}, I) where I is identity covariance
        X[t, :config.n_informative] = rng.normal(
            loc=means_inform[state_t, :],
            scale=1.0,  # Unit variance
            size=config.n_informative
        )
    
    # Generate noise features (zero mean across all states)
    if config.n_noise > 0:
        if config.correlated_noise:
            # Generate correlated Gaussian noise
            correlation = config.noise_correlation
            Sigma = np.eye(config.n_noise) * (1 - correlation) + correlation
            X[:, config.n_informative:] = rng.multivariate_normal(
                np.zeros(config.n_noise),
                Sigma,
                size=config.n_samples
            )
        else:
            # Independent Gaussian noise with zero mean, unit variance
            X[:, config.n_informative:] = rng.normal(
                loc=0.0,
                scale=1.0,
                size=(config.n_samples, config.n_noise)
            )
    
    # Create DataFrame with proper column names
    col_names = ([f'informative_{i+1}' for i in range(config.n_informative)] +
                 [f'noise_{i+1}' for i in range(config.n_noise)])
    X_df = pd.DataFrame(X, columns=col_names)
    
    return X_df, states, breakpoints


def generate_correlated_noise(n_samples: int,
                               n_features: int,
                               lambda_0: float,
                               correlation: float,
                               rng: np.random.RandomState) -> np.ndarray:
    """
    Generate correlated Poisson noise using Gaussian Copula (NORTA).
    
    Parameters:
    -----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of correlated noise features.
    lambda_0 : float
        Marginal Poisson rate for each feature.
    correlation : float
        Pairwise correlation between noise features.
    rng : np.random.RandomState
        Random number generator.
        
    Returns:
    --------
    np.ndarray
        Correlated count data of shape (n_samples, n_features).
    """
    # Create correlation matrix
    Sigma = np.eye(n_features) * (1 - correlation) + correlation
    
    # Generate correlated Gaussian latent variables
    Z = rng.multivariate_normal(np.zeros(n_features), Sigma, size=n_samples)
    
    # Transform to uniform [0,1] via standard normal CDF
    U = stats.norm.cdf(Z)
    
    # Transform to Poisson using inverse CDF
    # For Poisson, we use the ppf (percent point function)
    X = stats.poisson.ppf(U, lambda_0)
    
    return X


###############################################################################
# Evaluation Metrics
###############################################################################

def compute_bac_best_permutation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute balanced accuracy with best label permutation.
    
    Since clustering/segmentation algorithms can assign arbitrary labels to states,
    we need to find the permutation of predicted labels that maximizes agreement
    with true labels.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True state labels
    y_pred : np.ndarray
        Predicted state labels
        
    Returns:
    --------
    float
        Maximum balanced accuracy score across all permutations
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Get unique labels
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    # If different number of states, compute BAC with available states
    if len(unique_true) != len(unique_pred):
        # Fall back to standard BAC (will be suboptimal but won't crash)
        return balanced_accuracy_score(y_true, y_pred)
    
    K = len(unique_true)
    
    # Try all permutations of predicted labels
    best_bac = 0.0
    
    for perm in permutations(range(K)):
        # Create mapping from predicted labels to permuted labels
        label_map = {unique_pred[i]: unique_true[perm[i]] for i in range(K)}
        
        # Remap predicted labels
        y_pred_remapped = np.array([label_map.get(label, label) for label in y_pred])
        
        # Compute BAC for this permutation
        bac = balanced_accuracy_score(y_true, y_pred_remapped)
        
        if bac > best_bac:
            best_bac = bac
    
    return best_bac


def compute_persistence_reliability(true_states: np.ndarray,
                                    predicted_states: np.ndarray) -> Tuple[int, int, int]:
    """
    Compute persistence reliability: |J_est - J_true|.
    
    Parameters:
    -----------
    true_states : np.ndarray
        True state sequence.
    predicted_states : np.ndarray
        Predicted state sequence.
        
    Returns:
    --------
    n_jumps_true : int
        Number of true state changes.
    n_jumps_estimated : int
        Number of estimated state changes.
    persistence_error : int
        Absolute difference.
    """
    true_jumps = np.sum(true_states[:-1] != true_states[1:])
    pred_jumps = np.sum(predicted_states[:-1] != predicted_states[1:])
    
    return int(true_jumps), int(pred_jumps), int(abs(true_jumps - pred_jumps))


def compute_chamfer_distance(true_breakpoints: np.ndarray,
                             estimated_breakpoints: np.ndarray) -> float:
    """
    Compute Chamfer distance between true and estimated breakpoints.
    
    The Chamfer distance measures the average minimum distance from each point
    in one set to the nearest point in the other set, in both directions.
    
    CD(A, B) = mean(min_b∈B ||a - b||) + mean(min_a∈A ||b - a||)
    
    Lower values indicate better alignment of breakpoints.
    
    Parameters:
    -----------
    true_breakpoints : np.ndarray
        Indices of true state change points (breakpoints).
    estimated_breakpoints : np.ndarray
        Indices of estimated state change points.
        
    Returns:
    --------
    float
        Chamfer distance. Returns 0.0 if both sets are empty.
    """
    # Handle edge cases
    if len(true_breakpoints) == 0 and len(estimated_breakpoints) == 0:
        return 0.0
    if len(true_breakpoints) == 0:
        return float('inf')  # No true breakpoints but model found some
    if len(estimated_breakpoints) == 0:
        return float('inf')  # True breakpoints exist but none were found
    
    # Convert to numpy arrays if needed
    true_bp = np.asarray(true_breakpoints, dtype=float)
    est_bp = np.asarray(estimated_breakpoints, dtype=float)
    
    # For each true breakpoint, find distance to nearest estimated breakpoint
    distances_true_to_est = []
    for t in true_bp:
        min_dist = np.min(np.abs(est_bp - t))
        distances_true_to_est.append(min_dist)
    
    # For each estimated breakpoint, find distance to nearest true breakpoint
    distances_est_to_true = []
    for e in est_bp:
        min_dist = np.min(np.abs(true_bp - e))
        distances_est_to_true.append(min_dist)
    
    # Chamfer distance is the mean of both directional distances
    chamfer_dist = np.mean(distances_true_to_est) + np.mean(distances_est_to_true)
    
    return float(chamfer_dist)


def extract_breakpoints(state_sequence: np.ndarray) -> np.ndarray:
    """
    Extract breakpoint indices from a state sequence.
    
    A breakpoint occurs at index i when state[i] != state[i+1].
    
    Parameters:
    -----------
    state_sequence : np.ndarray
        Sequence of state labels.
        
    Returns:
    --------
    np.ndarray
        Indices where state changes occur.
    """
    # Find where state changes occur
    changes = np.where(state_sequence[:-1] != state_sequence[1:])[0]
    # The breakpoint is at the index before the change
    return changes


def compute_feature_selection_metrics(selected_features: List[int],
                                      n_informative: int,
                                      n_total: int) -> Dict[str, float]:
    """
    Compute feature selection quality metrics.
    
    Parameters:
    -----------
    selected_features : List[int]
        Indices of selected features (0-indexed).
    n_informative : int
        Number of truly informative features (first n_informative indices).
    n_total : int
        Total number of features.
        
    Returns:
    --------
    dict
        Dictionary with precision, recall, f1, and n_selected_noise.
    """
    # Ground truth: first n_informative features are informative
    true_informative = set(range(n_informative))
    selected_set = set(selected_features)
    
    # True positives, false positives, false negatives
    tp = len(selected_set & true_informative)
    fp = len(selected_set - true_informative)
    fn = len(true_informative - selected_set)
    
    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_selected_noise': fp
    }


def compute_poisson_deviance(y_true: np.ndarray,
                             lambda_pred: np.ndarray,
                             tol: float = 1e-10) -> float:
    """
    Compute Poisson deviance.
    
    D = 2 * Σ [y * log(y/λ) - (y - λ)]
    
    Parameters:
    -----------
    y_true : np.ndarray
        True count data.
    lambda_pred : np.ndarray
        Predicted Poisson rates.
    tol : float
        Small constant to avoid log(0).
        
    Returns:
    --------
    float
        Poisson deviance.
    """
    y_true = np.asarray(y_true)
    lambda_pred = np.maximum(lambda_pred, tol)
    
    # Handle y=0 case: lim_{y→0} y*log(y/λ) = 0
    mask = y_true > 0
    deviance = np.zeros_like(y_true, dtype=float)
    
    deviance[mask] = y_true[mask] * np.log(y_true[mask] / lambda_pred[mask])
    deviance -= (y_true - lambda_pred)
    
    return 2 * np.sum(deviance)


def compute_selection_stability(model_class,
                                X: np.ndarray,
                                model_params: Dict,
                                n_bootstrap: int = 50,
                                block_size: int = 10,
                                random_state: Optional[int] = None) -> float:
    """
    Compute feature selection stability using block bootstrap.
    
    Measures the robustness of feature selection using Jaccard similarity
    across bootstrap resamples with temporal dependence preservation.
    
    Parameters:
    -----------
    model_class : class
        Model class (e.g., SparseJumpModel).
    X : np.ndarray
        Input data.
    model_params : dict
        Parameters to pass to model constructor.
    n_bootstrap : int
        Number of bootstrap samples.
    block_size : int
        Size of blocks for block bootstrap.
    random_state : int, optional
        Random seed.
        
    Returns:
    --------
    float
        Mean Jaccard similarity across bootstrap pairs.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(X)
    selected_features_list = []
    
    for _ in range(n_bootstrap):
        # Block bootstrap
        n_blocks = n_samples // block_size
        block_starts = rng.choice(n_samples - block_size + 1, size=n_blocks, replace=True)
        indices = []
        for start in block_starts:
            indices.extend(range(start, min(start + block_size, n_samples)))
        indices = indices[:n_samples]  # Trim to original length
        
        X_boot = X[indices]
        
        # Fit model
        model = model_class(**model_params)
        model.fit(X_boot)
        
        # Get selected features (non-zero weights)
        if hasattr(model, 'feat_weights'):
            selected = np.where(model.feat_weights > 0)[0].tolist()
            selected_features_list.append(set(selected))
    
    # Compute pairwise Jaccard similarities
    similarities = []
    for i in range(len(selected_features_list)):
        for j in range(i + 1, len(selected_features_list)):
            set_i = selected_features_list[i]
            set_j = selected_features_list[j]
            if len(set_i | set_j) > 0:
                jaccard = len(set_i & set_j) / len(set_i | set_j)
                similarities.append(jaccard)
    
    return np.mean(similarities) if similarities else 0.0


###############################################################################
# Helper Functions
###############################################################################

def get_selected_features(model) -> List[int]:
    """
    Extract selected features from a fitted model.
    
    Parameters:
    -----------
    model : fitted model
        Model with feat_weights attribute (SparseJumpModel).
        
    Returns:
    --------
    List[int]
        Indices of features with non-zero weights.
    """
    if hasattr(model, 'feat_weights'):
        weights = model.feat_weights.values if hasattr(model.feat_weights, 'values') else model.feat_weights
        return np.where(weights > 1e-10)[0].tolist()
    else:
        # Non-sparse model: all features selected
        return list(range(model.centers_.shape[1]))


def split_train_validation(X: pd.DataFrame,
                           states: np.ndarray,
                           val_size: int = 100) -> Tuple:
    """
    Split data into training and validation sets.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Full dataset.
    states : np.ndarray
        True states.
    val_size : int
        Size of validation set (from the end).
        
    Returns:
    --------
    X_train, X_val, states_train, states_val
    """
    split_point = len(X) - val_size
    
    X_train = X.iloc[:split_point]
    X_val = X.iloc[split_point:]
    states_train = states[:split_point]
    states_val = states[split_point:]
    
    return X_train, X_val, states_train, states_val
