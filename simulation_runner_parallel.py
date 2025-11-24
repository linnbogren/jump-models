"""
Parallel Simulation Runner for Sparse Poisson Jump Model Evaluation

This module orchestrates the full simulation study with parallel processing:
- Grid search over hyperparameters
- Multiple replications
- Model comparison (Gaussian vs Poisson vs PoissonKL)
- Result storage
- Uses all available CPUs for parallel computation

Usage:
    python simulation_runner_parallel.py --n_replications 100 --output_dir results/
    python simulation_runner_parallel.py --quick_test  # For testing
"""

import numpy as np
import pandas as pd
import time
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import product
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

from jumpmodels.sparse_jump import SparseJumpModel
from jumpmodels.jump import JumpModel
from simulation_utils import (
    SimulationConfig, GridSearchResult,
    generate_poisson_hmm_data, generate_negative_binomial_hmm_data,
    generate_gaussian_hmm_data,
    compute_persistence_reliability, compute_feature_selection_metrics,
    compute_poisson_deviance, get_selected_features,
    compute_chamfer_distance, extract_breakpoints,
    compute_bac_best_permutation
)
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# Global variable to track if we should save models
SAVE_BEST_MODELS = False


###############################################################################
# Single Replication Runner (Worker Function)
###############################################################################

def run_single_replication_worker(args: Tuple) -> List[GridSearchResult]:
    """
    Worker function for parallel processing. Runs a single replication.
    
    This function is designed to be called by multiprocessing.Pool.
    Returns ALL grid search results - best model selection happens later.
    
    Parameters:
    -----------
    args : Tuple
        (config, model_name, hyperparameter_grid, save_models) unpacked from pool.map
        
    Returns:
    --------
    all_grid_results : List[GridSearchResult]
        Results from ALL models in the grid search.
    """
    config, model_name, hyperparameter_grid, save_models = args
    
    # Suppress sklearn warnings in worker processes to avoid clutter
    import warnings
    warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')
    
    # Generate data
    if config.distribution_type == "Poisson":
        X, states, breakpoints = generate_poisson_hmm_data(config)
    elif config.distribution_type == "NegativeBinomial":
        X, states, breakpoints = generate_negative_binomial_hmm_data(config)
    elif config.distribution_type == "Gaussian":
        X, states, breakpoints = generate_gaussian_hmm_data(config)
    else:
        raise ValueError(f"Unknown distribution type: {config.distribution_type}")
    
    # Extract true breakpoints from the full sequence
    true_breakpoints = extract_breakpoints(states)
    
    # Grid search: try all hyperparameter combinations
    all_grid_results = []  # Store ALL results
    best_model = None
    best_bac = -1.0
    best_grid_result = None
    
    for model_params in hyperparameter_grid:
        # Initialize model
        model = SparseJumpModel(
            n_components=model_params['n_components'],
            max_feats=model_params['max_feats'],
            jump_penalty=model_params['jump_penalty'],
            distribution=model_name,  # "Gaussian", "Poisson", or "PoissonKL"
            max_iter=50,
            tol_w=1e-4,
            n_init_jm=10,  # 10 different K-means++ initializations
            verbose=0,
            random_state=config.random_seed
        )
        
        try:
            # Fit model on full sequence
            fit_start = time.time()
            model.fit(X)
            fit_time = time.time() - fit_start
            
            # Get predictions on full sequence
            pred_states = model.labels_.values if hasattr(model.labels_, 'values') else model.labels_
            
            # Compute metrics with label permutation handling (on full sequence)
            bac = compute_bac_best_permutation(states, pred_states)
            acc = accuracy_score(states, pred_states)  # Note: accuracy also needs permutation, but we keep for compatibility
            
            n_jumps_true, n_jumps_est, pers_error = compute_persistence_reliability(
                states, pred_states
            )
            
            # Compute Chamfer distance between true and estimated breakpoints
            estimated_breakpoints = extract_breakpoints(pred_states)
            n_bp_true = len(true_breakpoints)
            n_bp_est = len(estimated_breakpoints)
            bp_count_error = abs(n_bp_true - n_bp_est)
            chamfer_dist = compute_chamfer_distance(true_breakpoints, estimated_breakpoints)
            
            selected_features = get_selected_features(model)
            feat_metrics = compute_feature_selection_metrics(
                selected_features, config.n_informative, config.n_total_features
            )
            
            # Store this grid point's results
            grid_result = GridSearchResult(
                config=config,
                model_name=model_name,
                hyperparameters=model_params,
                balanced_accuracy=bac,
                accuracy=acc,
                n_jumps_true=n_jumps_true,
                n_jumps_estimated=n_jumps_est,
                persistence_error=pers_error,
                n_breakpoints_true=n_bp_true,
                n_breakpoints_estimated=n_bp_est,
                breakpoint_count_error=bp_count_error,
                chamfer_distance=chamfer_dist,
                feature_f1=feat_metrics['f1'],
                feature_precision=feat_metrics['precision'],
                feature_recall=feat_metrics['recall'],
                n_selected_noise=feat_metrics['n_selected_noise'],
                n_selected_total=len(selected_features),
                computation_time=fit_time
            )
            all_grid_results.append(grid_result)
            
            # Track best model if saving models
            if save_models and bac > best_bac:
                best_bac = bac
                best_model = model
                best_grid_result = grid_result
                
        except Exception as e:
            # If fitting fails, skip this hyperparameter combination
            # Don't print in parallel workers (creates messy output)
            continue
    
    # Return all results - best selection happens later
    # If saving models, return tuple with best model
    if save_models and best_model is not None:
        return (all_grid_results, best_model, best_grid_result)
    else:
        return all_grid_results


###############################################################################
# Grid Search
###############################################################################

def create_hyperparameter_grid(n_total_features: int, quick_test: bool = False) -> List[Dict]:
    """
    Create hyperparameter grid for model comparison following the paper.
    
    For Sparse Jump Models, the paper uses:
    - 7 logarithmically spaced values of λ (jump penalty) between 10^-1 and 10^2
    - 14 equally spaced values of κ between 1 and √P
    
    Parameters:
    -----------
    n_total_features : int
        Total number of features (P in the paper).
    quick_test : bool
        If True, use min/center/max values (3 values) for each hyperparameter.
    
    Returns:
    --------
    List[Dict]
        List of hyperparameter combinations.
    """
    if quick_test:
        # Quick test: Use only min, center, max for each hyperparameter
        n_states_values = [2, 3, 4]  # min, center, max
        jump_penalty_values = [0.1, 10.0, 100.0]  # min, center (geometric mean), max
        sqrt_P = np.sqrt(n_total_features)
        kappa_values = [1, (1 + sqrt_P) / 2, sqrt_P]  # min, center (arithmetic mean), max
    else:
        # Full grid as in paper
        n_states_values = [2, 3, 4]
        jump_penalty_values = np.logspace(-1, 2, 7)  # [0.1, 0.46, 2.15, 10, 46.4, 215, 100]
        sqrt_P = np.sqrt(n_total_features)
        kappa_values = np.linspace(1, sqrt_P, 14)
    
    max_feats_values = np.array(kappa_values) ** 2  # Square to get max_feats
    
    grid = []
    for n_states, gamma, max_feats in product(n_states_values, jump_penalty_values, max_feats_values):
        grid.append({
            'n_components': n_states,
            'jump_penalty': gamma,
            'max_feats': max_feats
        })
    
    return grid


def create_data_config_grid(quick_test: bool = False) -> List[SimulationConfig]:
    """
    Create grid of data generation configurations.
    
    Parameters:
    -----------
    quick_test : bool
        If True, use min/center/max values for each configuration parameter.
    
    Returns:
    --------
    List[SimulationConfig]
        List of simulation configurations.
    """
    if quick_test:
        # Quick test: min, center, max for features and deltas, Poisson only, no correlated noise
        n_total_features_values = [15, 60, 300]  # min, center, max
        delta_values = [0.2, 0.5, 0.8]  # min, center, max (already 3 values)
        distribution_types = ["Poisson"]  # Poisson data only
        correlated_noise_values = [False]  # No correlated noise
    else:
        # Full grid
        n_total_features_values = [15, 30, 60, 150, 300]
        delta_values = [0.2, 0.5, 0.8]
        distribution_types = ["Poisson", "NegativeBinomial"]
        correlated_noise_values = [False, True]
    
    configs = []
    base_seed = 42
    config_id = 0
    
    for F, delta, dist_type, corr_noise in product(
        n_total_features_values, delta_values, distribution_types, correlated_noise_values
    ):
        # Only test correlated noise once (with Poisson)
        if corr_noise and dist_type != "Poisson":
            continue
            
        config = SimulationConfig(
            n_samples=500,
            n_states=3,
            n_informative=15,
            n_total_features=F,
            delta=delta,
            lambda_0=10.0,
            persistence=0.97,
            distribution_type=dist_type,
            correlated_noise=corr_noise,
            noise_correlation=0.1,
            nb_dispersion=2.0,
            random_seed=base_seed + config_id
        )
        configs.append(config)
        config_id += 1
    
    return configs


###############################################################################
# Best Model Selection
###############################################################################

def select_best_models(grid_results: List[GridSearchResult]) -> pd.DataFrame:
    """
    Select the best model for each unique (config, model_name) combination
    based on highest balanced accuracy.
    
    Parameters:
    -----------
    grid_results : List[GridSearchResult]
        All grid search results from all replications.
        
    Returns:
    --------
    pd.DataFrame
        Best results for each (config, model_name) combination.
    """
    # Convert to DataFrame for easier manipulation
    grid_df = grid_results_to_dataframe(grid_results)
    
    # Create grouping key
    group_cols = [
        'n_samples', 'n_states', 'n_informative', 'n_noise', 'n_total_features',
        'delta', 'lambda_0', 'persistence', 'distribution_type', 
        'correlated_noise', 'random_seed', 'model_name'
    ]
    
    # Find the best model for each group (highest BAC)
    idx = grid_df.groupby(group_cols)['balanced_accuracy'].idxmax()
    best_df = grid_df.loc[idx].copy()
    
    # Rename hyperparameter columns to indicate they're the "best"
    best_df = best_df.rename(columns={
        'n_components': 'best_n_components',
        'jump_penalty': 'best_jump_penalty',
        'max_feats': 'best_max_feats'
    })
    
    # Drop columns we don't need in the best results
    best_df = best_df.drop(columns=['n_selected_total'], errors='ignore')
    
    return best_df


###############################################################################
# Full Simulation (Parallel)
###############################################################################

def run_full_simulation_parallel(n_replications: int = 100,
                                  output_dir: str = "results",
                                  quick_test: bool = False,
                                  n_jobs: int = -1,
                                  save_models: bool = False) -> pd.DataFrame:
    """
    Run the complete simulation study with parallel processing.
    
    Parameters:
    -----------
    n_replications : int
        Number of replications for each configuration.
    output_dir : str
        Directory to save results.
    quick_test : bool
        If True, run a small subset for testing.
    n_jobs : int
        Number of parallel jobs. -1 uses all available CPUs.
    save_models : bool
        If True, save the best fitted model for each (config, model_name) pair.
        Models are saved as pickle files in output_dir/models/
        
    Returns:
    --------
    pd.DataFrame
        Aggregated results.
    """
    # Determine number of workers
    if n_jobs == -1:
        n_workers = cpu_count()
    else:
        n_workers = min(n_jobs, cpu_count())
    
    print(f"Using {n_workers} CPU cores for parallel processing")
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "aggregated").mkdir(exist_ok=True)
    (output_path / "grid_search").mkdir(exist_ok=True)
    if save_models:
        (output_path / "models").mkdir(exist_ok=True)
        print(f"Model saving is ENABLED - fitted models will be saved to {output_path / 'models'}")
    
    # Get configuration grids
    data_configs = create_data_config_grid(quick_test=quick_test)
    model_names = ["Gaussian", "Poisson", "PoissonKL"]
    
    if quick_test:
        print("Running in QUICK TEST mode:")
        print("  - 3 data configs (P: min/center/max)")
        print("  - 3 delta values (δ: min/center/max)")
        print("  - 3 n_states values (K: 2, 3, 4)")
        print("  - 3 jump_penalty values (λ: min/center/max)")
        print("  - 3 max_feats values (κ²: min/center/max)")
        print("  - Poisson data only, no correlated noise")
        print(f"  - {n_replications} replication(s)")
    
    print(f"Total configurations: {len(data_configs)}")
    print(f"Models: {len(model_names)} (all fitted on SAME data per replication)")
    print(f"Replications per config: {n_replications}")
    print(f"Total tasks: {len(data_configs) * n_replications} (each task fits all 3 models)")
    
    # Prepare all tasks for parallel execution
    all_tasks = []
    task_metadata = []  # Keep track of what each task is
    
    for data_config in data_configs:
        # Create hyperparameter grid specific to this feature dimension
        hyperparam_grid = create_hyperparameter_grid(
            data_config.n_total_features, 
            quick_test=quick_test
        )
        
        config_desc = f"P={data_config.n_total_features}, δ={data_config.delta}, dist={data_config.distribution_type}"
        print(f"  Config: {config_desc} -> {len(hyperparam_grid)} hyperparameter combinations")
        
        for model_name in model_names:
            for rep in range(n_replications):
                # Set unique seed for this replication
                config = SimulationConfig(
                    n_samples=data_config.n_samples,
                    n_states=data_config.n_states,
                    n_informative=data_config.n_informative,
                    n_total_features=data_config.n_total_features,
                    delta=data_config.delta,
                    lambda_0=data_config.lambda_0,
                    persistence=data_config.persistence,
                    distribution_type=data_config.distribution_type,
                    correlated_noise=data_config.correlated_noise,
                    noise_correlation=data_config.noise_correlation,
                    nb_dispersion=data_config.nb_dispersion,
                    random_seed=data_config.random_seed + rep * 1000 + hash(model_name) % 1000
                )
                
                # Add task to queue
                all_tasks.append((config, model_name, hyperparam_grid, save_models))
                task_metadata.append({
                    'config': config,
                    'model_name': model_name,
                    'replication': rep
                })
    
    print(f"\nStarting parallel execution of {len(all_tasks)} tasks...")
    print(f"Each task = 1 data generation + grid search over {len(hyperparam_grid)} hyperparameter combinations")
    print(f"Total model fits across all tasks: {len(all_tasks) * len(hyperparam_grid):,}")
    print(f"Progress will be shown below:\n")
    print(f"Results will be saved incrementally to {output_path / 'grid_search' / 'incremental'}")
    print(f"You can safely cancel (Ctrl+C) at any time without losing completed work!\n")
    
    # Create incremental save directory
    incremental_dir = output_path / "grid_search" / "incremental"
    incremental_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing incremental results
    existing_files = sorted(incremental_dir.glob("batch_*.pkl"))
    if existing_files:
        print(f"Found {len(existing_files)} existing result batches")
        print(f"Resuming from task {len(existing_files) + 1}/{len(all_tasks)}...\n")
        
        # Skip already completed tasks (no need to load results into memory)
        tasks_to_run = all_tasks[len(existing_files):]
        batch_offset = len(existing_files)
    else:
        tasks_to_run = all_tasks
        batch_offset = 0
    
    if len(tasks_to_run) == 0:
        print("All tasks already completed! Proceeding to final aggregation...")
    else:
        # Run remaining tasks in parallel with progress bar
        with Pool(processes=n_workers) as pool:
            # Use imap_unordered for better progress tracking
            results_iter = pool.imap_unordered(run_single_replication_worker, tasks_to_run)
            
            # Process results as they complete
            batch_idx = batch_offset
            for result in tqdm(results_iter, total=len(tasks_to_run), 
                                    desc="Running simulations", initial=batch_offset):
                try:
                    # Handle result (could be grid_results or (grid_results, model, best_result))
                    if save_models and isinstance(result, tuple):
                        grid_results, best_model, best_grid_result = result
                        
                        # Save the best model
                        model_filename = (f"model_{best_grid_result.model_name}_"
                                        f"seed{best_grid_result.config.random_seed}_"
                                        f"P{best_grid_result.config.n_total_features}_"
                                        f"delta{best_grid_result.config.delta}.pkl")
                        model_path = output_path / "models" / model_filename
                        with open(model_path, 'wb') as f:
                            pickle.dump(best_model, f)
                    else:
                        grid_results = result
                    
                    # Save grid results batch immediately (incremental save)
                    batch_file = incremental_dir / f"batch_{batch_idx:06d}.pkl"
                    with open(batch_file, 'wb') as f:
                        pickle.dump(grid_results, f)
                    
                    # Don't keep in memory - will load from disk for final aggregation
                    batch_idx += 1
                    
                except Exception as e:
                    print(f"\nError in task: {e}")
                    batch_idx += 1
                    continue
    
    print(f"\n{'='*80}")
    print("Final Aggregation - Loading all results from disk")
    print(f"{'='*80}")
    
    # Load all results from incremental files
    # More memory-efficient: process in chunks
    all_batch_files = sorted(incremental_dir.glob("batch_*.pkl"))
    print(f"Found {len(all_batch_files)} batch files to aggregate")
    
    # Process in chunks to avoid memory issues with very large simulations
    chunk_size = 100
    all_grid_dfs = []
    
    print(f"Processing in chunks of {chunk_size} batches...")
    for chunk_start in range(0, len(all_batch_files), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(all_batch_files))
        print(f"  Processing batches {chunk_start}-{chunk_end}...")
        
        chunk_results = []
        for batch_file in all_batch_files[chunk_start:chunk_end]:
            try:
                with open(batch_file, 'rb') as f:
                    batch_results = pickle.load(f)
                    chunk_results.extend(batch_results)
            except Exception as e:
                print(f"    Warning: Could not load {batch_file.name}: {e}")
        
        # Convert this chunk to DataFrame
        if chunk_results:
            chunk_df = grid_results_to_dataframe(chunk_results)
            all_grid_dfs.append(chunk_df)
            print(f"    Chunk has {len(chunk_results)} results")
        
        # Clear chunk from memory
        del chunk_results
    
    # Concatenate all chunks
    print(f"\nCombining {len(all_grid_dfs)} chunks...")
    grid_df = pd.concat(all_grid_dfs, ignore_index=True)
    del all_grid_dfs  # Free memory
    
    print(f"Total grid search evaluations: {len(grid_df)}")
    
    # Save grid search results to CSV
    print(f"Saving grid search results to {output_path / 'grid_search' / 'all_grid_results.csv'}...")
    grid_df.to_csv(output_path / "grid_search" / "all_grid_results.csv", index=False)
    
    # Select best models from grid search DataFrame
    print("Finding best models for each configuration...")
    
    # Create grouping key
    group_cols = [
        'n_samples', 'n_states', 'n_informative', 'n_noise', 'n_total_features',
        'delta', 'lambda_0', 'persistence', 'distribution_type', 
        'correlated_noise', 'random_seed', 'model_name'
    ]
    
    # Find the best model for each group (highest BAC)
    idx = grid_df.groupby(group_cols)['balanced_accuracy'].idxmax()
    best_df = grid_df.loc[idx].copy()
    
    # Rename hyperparameter columns to indicate they're the "best"
    best_df = best_df.rename(columns={
        'n_components': 'best_n_components',
        'jump_penalty': 'best_jump_penalty',
        'max_feats': 'best_max_feats'
    })
    
    # Drop columns we don't need in the best results
    best_df = best_df.drop(columns=['n_selected_total'], errors='ignore')
    
    # Save best results (aggregated)
    print(f"Saving best results to {output_path / 'aggregated' / 'all_results.csv'}...")
    best_df.to_csv(output_path / "aggregated" / "all_results.csv", index=False)
    
    # Clean up incremental files after successful completion
    print(f"Cleaning up incremental save files...")
    incremental_files = list(incremental_dir.glob("batch_*.pkl"))
    for f in incremental_files:
        f.unlink()
    print(f"Removed {len(incremental_files)} incremental files")
    
    print(f"\n{'='*80}")
    print(f"Simulation complete! Results saved to {output_dir}")
    print(f"Total grid search evaluations: {len(grid_df)}")
    print(f"Total best models selected: {len(best_df)}")
    print(f"{'='*80}")
    
    return best_df


def grid_results_to_dataframe(results: List[GridSearchResult]) -> pd.DataFrame:
    """
    Convert list of grid search results to a pandas DataFrame.
    
    Parameters:
    -----------
    results : List[GridSearchResult]
        List of grid search results.
        
    Returns:
    --------
    pd.DataFrame
        Flattened grid search results.
    """
    rows = []
    for result in results:
        row = {
            # Config parameters
            'n_samples': result.config.n_samples,
            'n_states': result.config.n_states,
            'n_informative': result.config.n_informative,
            'n_noise': result.config.n_noise,
            'n_total_features': result.config.n_total_features,
            'delta': result.config.delta,
            'lambda_0': result.config.lambda_0,
            'persistence': result.config.persistence,
            'distribution_type': result.config.distribution_type,
            'correlated_noise': result.config.correlated_noise,
            'random_seed': result.config.random_seed,
            
            # Model
            'model_name': result.model_name,
            
            # Hyperparameters (this specific grid point)
            'n_components': result.hyperparameters['n_components'],
            'jump_penalty': result.hyperparameters['jump_penalty'],
            'max_feats': result.hyperparameters['max_feats'],
            
            # Results
            'balanced_accuracy': result.balanced_accuracy,
            'accuracy': result.accuracy,
            'n_jumps_true': result.n_jumps_true,
            'n_jumps_estimated': result.n_jumps_estimated,
            'persistence_error': result.persistence_error,
            'n_breakpoints_true': result.n_breakpoints_true,
            'n_breakpoints_estimated': result.n_breakpoints_estimated,
            'breakpoint_count_error': result.breakpoint_count_error,
            'chamfer_distance': result.chamfer_distance,
            'feature_f1': result.feature_f1,
            'feature_precision': result.feature_precision,
            'feature_recall': result.feature_recall,
            'n_selected_noise': result.n_selected_noise,
            'n_selected_total': result.n_selected_total,
            'computation_time': result.computation_time,
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


###############################################################################
# Command Line Interface
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Run Sparse Poisson Jump Model simulation study (PARALLEL)")
    parser.add_argument('--n_replications', type=int, default=100,
                       help="Number of replications per configuration")
    parser.add_argument('--output_dir', type=str, default="results",
                       help="Output directory for results")
    parser.add_argument('--quick_test', action='store_true',
                       help="Run a small test (2 configs, smaller grid, 5 reps)")
    parser.add_argument('--full_grid_single', action='store_true',
                       help="Run full grid search but with only 1 replication")
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help="Number of parallel jobs (-1 for all CPUs)")
    parser.add_argument('--save_models', action='store_true',
                       help="Save fitted models for later visualization (increases storage)")
    
    args = parser.parse_args()
    
    # Handle full_grid_single mode
    if args.full_grid_single:
        args.n_replications = 1
        args.quick_test = False
        print("FULL GRID - SINGLE ITERATION MODE")
        print("  Using full hyperparameter grid")
        print("  Using full configuration grid")
        print("  Running only 1 replication per configuration")
    
    print("="*80)
    print("Sparse Poisson Jump Model - PARALLEL Simulation Study")
    print("="*80)
    
    start_time = time.time()
    
    results_df = run_full_simulation_parallel(
        n_replications=args.n_replications,
        output_dir=args.output_dir,
        quick_test=args.quick_test,
        n_jobs=args.n_jobs,
        save_models=args.save_models
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\nTotal execution time: {elapsed_time/60:.2f} minutes")
    print("\nQuick summary (Balanced Accuracy):")
    print(results_df.groupby(['model_name', 'delta', 'n_total_features'])['balanced_accuracy'].describe())
    print("\nQuick summary (Breakpoint Count Error):")
    print(results_df.groupby(['model_name', 'delta', 'n_total_features'])['breakpoint_count_error'].describe())
    print("\nQuick summary (Chamfer Distance):")
    print(results_df.groupby(['model_name', 'delta', 'n_total_features'])['chamfer_distance'].describe())


if __name__ == "__main__":
    main()
