"""
Visualize Extreme Cases: Largest BAC and Chamfer Differences

This script analyzes all simulation results to find:
1. The configuration with the largest BAC difference between models
2. The configuration with the largest Chamfer distance difference between models

Then generates detailed visualizations for these extreme cases.

If saved model files are available (from running simulation with --save_models),
they will be loaded for exact reproduction. Otherwise, models will be refitted
with stored hyperparameters (which may produce different results due to stochastic
K-means++ initialization with n_init_jm=10).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
from itertools import permutations
from jumpmodels.sparse_jump import SparseJumpModel
from simulation_utils import (
    SimulationConfig,
    generate_poisson_hmm_data,
    extract_breakpoints,
    compute_chamfer_distance
)
from sklearn.metrics import balanced_accuracy_score


def compute_bac_best_permutation(y_true, y_pred, return_permuted=False):
    """
    Compute balanced accuracy with best label permutation.
    
    Parameters:
    -----------
    y_true : array-like
        True state labels
    y_pred : array-like
        Predicted state labels
    return_permuted : bool
        If True, also return the permuted predictions
        
    Returns:
    --------
    float or tuple
        If return_permuted=False: best BAC score
        If return_permuted=True: (best BAC score, permuted predictions)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    unique_true = np.unique(y_true)
    unique_pred = np.unique(y_pred)
    
    if len(unique_true) != len(unique_pred):
        if return_permuted:
            return balanced_accuracy_score(y_true, y_pred), y_pred
        return balanced_accuracy_score(y_true, y_pred)
    
    K = len(unique_true)
    best_bac = 0.0
    best_y_pred_perm = y_pred.copy()
    
    for perm in permutations(range(K)):
        mapping = {unique_pred[i]: perm[i] for i in range(K)}
        y_pred_perm = np.array([mapping[label] for label in y_pred])
        bac = balanced_accuracy_score(y_true, y_pred_perm)
        
        if bac > best_bac:
            best_bac = bac
            best_y_pred_perm = y_pred_perm
    
    if return_permuted:
        return best_bac, best_y_pred_perm
    return best_bac


# Set style
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 24,
    'figure.dpi': 300,
})

# Create output directory
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Finding Extreme Cases: Largest BAC and Chamfer Differences")
print("="*80)

###############################################################################
# Load and Analyze All Results
###############################################################################

print("\nLoading simulation results...")
results_file = Path("results/aggregated/all_results.csv")

if not results_file.exists():
    print(f"Error: Results file not found at {results_file}")
    print("Please run simulations first!")
    exit(1)

df = pd.read_csv(results_file)
print(f"Loaded {len(df)} results")
print(f"Models: {df['model_name'].unique()}")

# Group by configuration (excluding random_seed and model_name)
config_cols = ['n_samples', 'n_states', 'n_informative', 'n_noise', 'n_total_features',
               'delta', 'lambda_0', 'persistence', 'distribution_type', 'correlated_noise']

# Compute differences for each configuration
print("\nAnalyzing differences across configurations...")

extreme_cases = []

for config_key, group in df.groupby(config_cols):
    # Get one example of each model for this config
    models_data = {}
    for model_name in ['Gaussian', 'Poisson', 'PoissonKL']:
        model_df = group[group['model_name'] == model_name]
        if len(model_df) > 0:
            models_data[model_name] = model_df.iloc[0]
    
    if len(models_data) != 3:
        continue
    
    # Compute BAC differences
    bac_values = {name: data['balanced_accuracy'] for name, data in models_data.items()}
    max_bac_diff = max(bac_values.values()) - min(bac_values.values())
    best_bac_model = max(bac_values.items(), key=lambda x: x[1])[0]
    worst_bac_model = min(bac_values.items(), key=lambda x: x[1])[0]
    
    # Compute Chamfer differences
    chamfer_values = {name: data['chamfer_distance'] for name, data in models_data.items()}
    max_chamfer_diff = max(chamfer_values.values()) - min(chamfer_values.values())
    best_chamfer_model = min(chamfer_values.items(), key=lambda x: x[1])[0]
    worst_chamfer_model = max(chamfer_values.items(), key=lambda x: x[1])[0]
    
    extreme_cases.append({
        'config': dict(zip(config_cols, config_key)),
        'random_seed': models_data['Gaussian']['random_seed'],
        'max_bac_diff': max_bac_diff,
        'best_bac_model': best_bac_model,
        'worst_bac_model': worst_bac_model,
        'bac_values': bac_values,
        'max_chamfer_diff': max_chamfer_diff,
        'best_chamfer_model': best_chamfer_model,
        'worst_chamfer_model': worst_chamfer_model,
        'chamfer_values': chamfer_values,
        'models_data': models_data
    })

# Find extreme cases
extreme_cases_df = pd.DataFrame(extreme_cases)
max_bac_diff_case = extreme_cases_df.loc[extreme_cases_df['max_bac_diff'].idxmax()]
max_chamfer_diff_case = extreme_cases_df.loc[extreme_cases_df['max_chamfer_diff'].idxmax()]

print(f"\n{'='*80}")
print("EXTREME CASE #1: Largest BAC Difference")
print(f"{'='*80}")
print(f"BAC Difference: {max_bac_diff_case['max_bac_diff']:.4f}")
print(f"Best Model: {max_bac_diff_case['best_bac_model']} (BAC = {max_bac_diff_case['bac_values'][max_bac_diff_case['best_bac_model']]:.4f})")
print(f"Worst Model: {max_bac_diff_case['worst_bac_model']} (BAC = {max_bac_diff_case['bac_values'][max_bac_diff_case['worst_bac_model']]:.4f})")
print(f"\nConfiguration:")
for key, value in max_bac_diff_case['config'].items():
    print(f"  {key}: {value}")
print(f"  random_seed: {max_bac_diff_case['random_seed']}")

print(f"\n{'='*80}")
print("EXTREME CASE #2: Largest Chamfer Distance Difference")
print(f"{'='*80}")
print(f"Chamfer Difference: {max_chamfer_diff_case['max_chamfer_diff']:.2f}")
print(f"Best Model: {max_chamfer_diff_case['best_chamfer_model']} (Chamfer = {max_chamfer_diff_case['chamfer_values'][max_chamfer_diff_case['best_chamfer_model']]:.2f})")
print(f"Worst Model: {max_chamfer_diff_case['worst_chamfer_model']} (Chamfer = {max_chamfer_diff_case['chamfer_values'][max_chamfer_diff_case['worst_chamfer_model']]:.2f})")
print(f"\nConfiguration:")
for key, value in max_chamfer_diff_case['config'].items():
    print(f"  {key}: {value}")
print(f"  random_seed: {max_chamfer_diff_case['random_seed']}")

###############################################################################
# Function to Visualize a Case
###############################################################################

def load_or_fit_model(model_name, distribution, config, hyperparams, X, model_random_seed):
    """
    Load saved model if available, otherwise fit a new one.
    
    Parameters:
    -----------
    model_name : str
        Name of the model ('Gaussian', 'Poisson', 'PoissonKL')
    distribution : str
        Distribution type for the model
    config : SimulationConfig
        Configuration used to generate the data
    hyperparams : dict
        Dictionary with 'n_components', 'jump_penalty', 'max_feats'
    X : DataFrame
        Feature data to fit on
    model_random_seed : int
        The random seed that was actually used for this model (from CSV)
        
    Returns:
    --------
    model : SparseJumpModel
        The fitted model (either loaded or newly fitted)
    was_loaded : bool
        True if model was loaded from file, False if fitted
    """
    # Construct model filename using the actual model seed from CSV
    models_dir = Path("results/models")
    model_filename = (f"model_{model_name}_seed{model_random_seed}_"
                     f"P{config.n_total_features}_delta{config.delta}.pkl")
    model_path = models_dir / model_filename
    
    # Try to load saved model
    if model_path.exists():
        print(f"  Loading saved model from {model_filename}...")
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model, True
        except Exception as e:
            print(f"  Warning: Failed to load model ({e}), will refit instead")
    
    # If not found or failed to load, fit new model
    print(f"  Fitting new model (no saved model found at {model_filename})...")
    model = SparseJumpModel(
        n_components=hyperparams['n_components'],
        max_feats=hyperparams['max_feats'],
        jump_penalty=hyperparams['jump_penalty'],
        distribution=distribution,
        max_iter=50,
        tol_w=1e-4,
        n_init_jm=10,
        verbose=0,
        random_state=model_random_seed
    )
    
    model.fit(X)
    return model, False


def visualize_extreme_case(case, case_name, output_suffix):
    """Generate comprehensive visualizations for an extreme case."""
    
    print(f"\n{'='*80}")
    print(f"Generating Visualizations: {case_name}")
    print(f"{'='*80}")
    
    # Extract configuration (use first model's config as template)
    config_dict = case['config']
    base_config = SimulationConfig(
        n_samples=int(config_dict['n_samples']),
        n_states=int(config_dict['n_states']),
        n_informative=int(config_dict['n_informative']),
        n_noise=int(config_dict['n_noise']),
        n_total_features=int(config_dict['n_total_features']),
        delta=float(config_dict['delta']),
        lambda_0=float(config_dict['lambda_0']),
        persistence=float(config_dict['persistence']),
        distribution_type=str(config_dict['distribution_type']),
        correlated_noise=bool(config_dict['correlated_noise']),
        random_seed=int(case['random_seed'])  # Base seed (from data generation)
    )
    
    # Get hyperparameters from stored results
    example_data = case['models_data']['Gaussian']
    hyperparams = {
        'n_components': int(example_data['best_n_components']),
        'jump_penalty': float(example_data['best_jump_penalty']),
        'max_feats': float(example_data['best_max_feats'])
    }
    
    print(f"Model hyperparameters: K={hyperparams['n_components']}, "
          f"λ={hyperparams['jump_penalty']}, κ²={hyperparams['max_feats']}")
    
    # Generate data and fit/load models for each model type
    # Note: Each model in the original simulation used its own random seed for data generation
    models = {}
    predictions = {}
    performance = {}
    models_loaded = {}
    data_cache = {}  # Cache generated data for each seed
    
    model_specs = {
        'Gaussian': {'distribution': 'Gaussian', 'color': '#1f77b4'},
        'Poisson': {'distribution': 'Poisson', 'color': '#ff7f0e'},
        'PoissonKL': {'distribution': 'PoissonKL', 'color': '#2ca02c'}
    }
    
    # First pass: Load/fit all models and generate their respective data
    for model_name, spec in model_specs.items():
        print(f"\n{model_name}:")
        
        # Get the actual random seed used for this specific model from the CSV
        model_random_seed = int(case['models_data'][model_name]['random_seed'])
        
        # Generate data with this model's seed (each model had different data in simulation!)
        if model_random_seed not in data_cache:
            config = SimulationConfig(
                n_samples=base_config.n_samples,
                n_states=base_config.n_states,
                n_informative=base_config.n_informative,
                n_noise=base_config.n_noise,
                n_total_features=base_config.n_total_features,
                delta=base_config.delta,
                lambda_0=base_config.lambda_0,
                persistence=base_config.persistence,
                distribution_type=base_config.distribution_type,
                correlated_noise=base_config.correlated_noise,
                random_seed=model_random_seed
            )
            X, states, breakpoints = generate_poisson_hmm_data(config)
            data_cache[model_random_seed] = (X, states, breakpoints)
        else:
            X, states, breakpoints = data_cache[model_random_seed]
        
        import time
        start_time = time.time()
        model, was_loaded = load_or_fit_model(
            model_name, spec['distribution'], base_config, hyperparams, X, model_random_seed
        )
        load_or_fit_time = time.time() - start_time
        
        models[model_name] = model
        models_loaded[model_name] = was_loaded
        pred_states = model.labels_.values if hasattr(model.labels_, 'values') else model.labels_
        
        # Extract true breakpoints
        true_breakpoints = extract_breakpoints(states)
        
        # Get best permutation for BAC matching (on the data used for this model)
        bac, pred_states_permuted = compute_bac_best_permutation(states, pred_states, return_permuted=True)
        
        predictions[model_name] = pred_states_permuted  # Store permuted predictions
        
        pred_breakpoints = extract_breakpoints(pred_states_permuted)
        chamfer = compute_chamfer_distance(true_breakpoints, pred_breakpoints)
        
        performance[model_name] = {
            'BAC': bac,
            'Chamfer': chamfer,
            'Time': load_or_fit_time,
            'Breakpoints': pred_breakpoints,
            'N_Breakpoints': len(pred_breakpoints),
            'Loaded': was_loaded,
            'true_breakpoints': true_breakpoints,
            'states': states,
            'X': X
        }
        
        status = "✓ LOADED" if was_loaded else "⚠ REFITTED"
        print(f"  {status}: BAC={bac:.4f}, Chamfer={chamfer:.2f}, Time={load_or_fit_time:.2f}s")
    
    # Check if any models were refitted
    any_refitted = not all(models_loaded.values())
    if any_refitted:
        print(f"\n⚠ WARNING: Some models were refitted (not loaded from saved files).")
        print(f"  Results may differ from stored values due to stochastic initialization.")
        print(f"  To reproduce exact results, run simulation with --save_models flag.")
    else:
        print(f"\n✓ All models loaded from saved files - results match stored values exactly!")
    
    # For visualization, use the data from the first model (Gaussian) as reference
    # This is just for display purposes - each model was evaluated on its own data above
    X_viz = performance['Gaussian']['X']
    states_viz = performance['Gaussian']['states']
    true_breakpoints_viz = performance['Gaussian']['true_breakpoints']
    
    ###########################################################################
    # Figure 1: Feature Weights Comparison
    ###########################################################################
    
    print(f"\nCreating feature weights visualization...")
    
    informative_features = [col for col in X_viz.columns if 'informative' in col]
    noise_features = [col for col in X_viz.columns if 'noise' in col]
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    for idx, model_name in enumerate(['Gaussian', 'Poisson', 'PoissonKL']):
        ax = axes[idx]
        model = models[model_name]
        
        # For PoissonKL, use sqrt of weights for comparable scale
        if model_name == 'PoissonKL':
            weights = np.sqrt(model.feat_weights)
        else:
            weights = model.feat_weights
        
        # Separate informative and noise features
        informative_weights = weights[informative_features].values
        noise_weights = weights[noise_features].values
        
        # Create positions for bars
        n_informative = len(informative_weights)
        n_noise = len(noise_weights)
        
        x_informative = np.arange(n_informative)
        x_noise = np.arange(n_informative, n_informative + n_noise)
        
        # Plot bars (without edge color)
        ax.bar(x_informative, informative_weights, color='#2ecc71', alpha=0.8,
               edgecolor='none', label='Informative')
        ax.bar(x_noise, noise_weights, color='#e74c3c', alpha=0.8,
               edgecolor='none', label='Noise')
        
        # Add horizontal lines for mean weights
        if len(informative_weights) > 0:
            mean_informative = informative_weights.mean()
            ax.axhline(y=mean_informative, color='#27ae60', linestyle='--', linewidth=2.5,
                       alpha=0.7, label=f'Inf. Mean: {mean_informative:.3f}')
        
        if len(noise_weights) > 0:
            mean_noise = noise_weights.mean()
            ax.axhline(y=mean_noise, color='#c0392b', linestyle='--', linewidth=2.5,
                       alpha=0.7, label=f'Noise Mean: {mean_noise:.3f}')
        
        # Formatting
        bac = performance[model_name]['BAC']
        chamfer = performance[model_name]['Chamfer']
        title_text = f'{model_name}'
        if model_name == 'PoissonKL':
            title_text += ' (sqrt weights)'
        title_text += f'\nBAC={bac:.4f}, Ch={chamfer:.1f}'
        
        ax.set_title(title_text, fontsize=24, fontweight='bold', pad=15)
        ax.set_xlabel('Feature Index', fontsize=20, fontweight='bold')
        ax.set_ylabel('Feature Weight', fontsize=20, fontweight='bold')
        ax.legend(fontsize=16, frameon=True, shadow=True, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=16)
        
        # Add vertical separator
        if n_informative > 0 and n_noise > 0:
            ax.axvline(x=n_informative - 0.5, color='black', linestyle=':',
                      linewidth=2, alpha=0.5)
            ax.text(n_informative/2, ax.get_ylim()[1]*0.95, 'Informative',
                   ha='center', fontsize=14, fontweight='bold')
            ax.text(n_informative + n_noise/2, ax.get_ylim()[1]*0.95, 'Noise',
                   ha='center', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'{case_name}\nΔBAC={case["max_bac_diff"]:.4f}, ΔChamfer={case["max_chamfer_diff"]:.2f}',
                 fontsize=26, fontweight='bold', y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / f"extreme_{output_suffix}_weights.jpg",
                dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    
    print(f"  ✓ Saved: extreme_{output_suffix}_weights.jpg")
    
    ###########################################################################
    # Figure 2: Stacked States
    ###########################################################################
    
    print(f"Creating stacked states visualization...")
    
    # Select best informative feature (from training data)
    avg_weights = {}
    for feature in informative_features:
        avg_weight = np.mean([models[m].feat_weights[feature] for m in ['Gaussian', 'Poisson', 'PoissonKL']])
        avg_weights[feature] = avg_weight
    
    best_informative = max(avg_weights.items(), key=lambda x: x[1])[0]
    
    n_models = 3
    n_panels = 2 + n_models
    
    fig, axes = plt.subplots(n_panels, 1, figsize=(22, 3 * n_panels), sharex=True)
    
    fig.suptitle(f'{case_name}: {best_informative}\nΔBAC={case["max_bac_diff"]:.4f}, ΔChamfer={case["max_chamfer_diff"]:.2f}',
                 fontsize=26, fontweight='bold', y=0.995)
    
    panel_idx = 0
    
    # Panel 0: Time series (full sequence from visualization data)
    ax = axes[panel_idx]
    ax.plot(X_viz.index, X_viz[best_informative], color='#2c3e50', linewidth=2.5, alpha=0.8)
    ax.set_ylabel('Count', fontsize=18, fontweight='bold')
    ax.set_title(f'{best_informative}', fontsize=22, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='y', labelsize=16)
    panel_idx += 1
    
    # Color map for states (full sequence)
    n_states = len(np.unique(states_viz))
    cmap = plt.colormaps.get_cmap('Set2')
    color_map = {i: cmap(i / max(n_states - 1, 1)) for i in range(n_states)}
    
    def plot_state_bars(ax, labels, label_text, bac=None, chamfer=None, n_breakpoints=None):
        """Plot horizontal state bars."""
        labels_array = np.asarray(labels)
        current_state = labels_array[0]
        start_idx = 0
        
        for t in range(1, len(labels_array) + 1):
            if t == len(labels_array) or labels_array[t] != current_state:
                y_pos = current_state + 1
                ax.barh(y=y_pos, width=t - start_idx, left=start_idx,
                       height=0.8, color=color_map[current_state],
                       edgecolor='white', linewidth=1.5, align='center')
                
                if t < len(labels_array):
                    current_state = labels_array[t]
                    start_idx = t
        
        ax.set_ylabel('State', fontsize=16, fontweight='bold')
        ax.set_title(label_text, fontsize=20, fontweight='bold', pad=10, loc='left')
        
        # Add metrics
        if bac is not None and chamfer is not None:
            text = f'BAC={bac:.3f}\nCh={chamfer:.1f}'
            if n_breakpoints is not None:
                n_true_breakpoints = len(true_breakpoints)
                bp_error = abs(n_true_breakpoints - n_breakpoints)
                text += f'\nBP={n_breakpoints} (Δ={bp_error})'
            ax.text(1.02, 0.5, text,
                    transform=ax.transAxes, fontsize=18, fontweight='bold',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9,
                            edgecolor='black', linewidth=2))
        elif n_breakpoints is not None:
            # For true states, just show breakpoint count
            text = f'BP={n_breakpoints}'
            ax.text(1.02, 0.5, text,
                    transform=ax.transAxes, fontsize=18, fontweight='bold',
                    verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9,
                            edgecolor='black', linewidth=2))
        
        ax.set_ylim(0.5, n_states + 0.5)
        ax.set_yticks(range(1, n_states + 1))
        ax.set_yticklabels([f'{i}' for i in range(n_states)], fontsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.2, axis='x', linestyle='--')
    
    # Panel 1: True states (from visualization data)
    ax = axes[panel_idx]
    plot_state_bars(ax, states_viz, 'True States', n_breakpoints=len(true_breakpoints_viz))
    panel_idx += 1
    
    # Panels 2-4: Model predictions
    for model_name in ['Gaussian', 'Poisson', 'PoissonKL']:
        ax = axes[panel_idx]
        pred_states = predictions[model_name]
        bac = performance[model_name]['BAC']
        chamfer = performance[model_name]['Chamfer']
        n_bp = performance[model_name]['N_Breakpoints']
        plot_state_bars(ax, pred_states, f'{model_name}', bac=bac, chamfer=chamfer, n_breakpoints=n_bp)
        panel_idx += 1
    
    axes[-1].set_xlabel('Time Step', fontsize=20, fontweight='bold')
    axes[-1].tick_params(labelsize=16)
    
    plt.tight_layout(rect=[0, 0.01, 1, 0.995])
    plt.subplots_adjust(hspace=0.25)
    plt.savefig(output_dir / f"extreme_{output_suffix}_stacked.jpg",
                dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    
    print(f"  ✓ Saved: extreme_{output_suffix}_stacked.jpg")
    
    return performance

###############################################################################
# Generate Visualizations for Both Extreme Cases
###############################################################################

perf1 = visualize_extreme_case(
    max_bac_diff_case,
    "Largest BAC Difference",
    "max_bac_diff"
)

perf2 = visualize_extreme_case(
    max_chamfer_diff_case,
    "Largest Chamfer Difference",
    "max_chamfer_diff"
)

###############################################################################
# Summary
###############################################################################

print(f"\n{'='*80}")
print("Summary")
print(f"{'='*80}")

print(f"\nExtreme Case #1: Largest BAC Difference ({max_bac_diff_case['max_bac_diff']:.4f})")
print(f"  Generated files:")
print(f"    - extreme_max_bac_diff_weights.jpg")
print(f"    - extreme_max_bac_diff_stacked.jpg")

print(f"\nExtreme Case #2: Largest Chamfer Difference ({max_chamfer_diff_case['max_chamfer_diff']:.2f})")
print(f"  Generated files:")
print(f"    - extreme_max_chamfer_diff_weights.jpg")
print(f"    - extreme_max_chamfer_diff_stacked.jpg")

print(f"\nAll figures saved to: {output_dir}/")
print("\nDone!")

