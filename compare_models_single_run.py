"""
Single Run Model Comparison Script

Runs a single simulation with mid-range parameters (no grid search) and compares
Gaussian, Poisson, and PoissonKL models on Poisson-generated data.

Generates visualizations showing:
- Time series with true and predicted breakpoints
- Feature weights comparison
- Performance metrics (BAC, Chamfer distance)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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


def compute_bac_best_permutation(y_true, y_pred):
    """
    Compute balanced accuracy with best label permutation.
    
    Since clustering/segmentation algorithms can assign arbitrary labels to states,
    we need to find the permutation of predicted labels that maximizes agreement
    with true labels.
    
    Parameters:
    -----------
    y_true : array-like
        True state labels
    y_pred : array-like
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
    
    # If different number of states, something went wrong
    if len(unique_true) != len(unique_pred):
        print(f"Warning: Different number of states (true: {len(unique_true)}, pred: {len(unique_pred)})")
        return balanced_accuracy_score(y_true, y_pred)
    
    K = len(unique_true)
    
    # Try all permutations of predicted labels
    best_bac = 0.0
    best_perm = None
    
    for perm in permutations(range(K)):
        # Create mapping: pred_label -> permuted_label
        mapping = {unique_pred[i]: perm[i] for i in range(K)}
        
        # Apply permutation
        y_pred_perm = np.array([mapping[label] for label in y_pred])
        
        # Compute BAC for this permutation
        bac = balanced_accuracy_score(y_true, y_pred_perm)
        
        if bac > best_bac:
            best_bac = bac
            best_perm = perm
    
    return best_bac


# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 26,
    'figure.dpi': 300,
})

# Create output directory
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Model Comparison: Gaussian vs Poisson vs PoissonKL (10 iterations)")
print("="*80)

###############################################################################
# Configuration - Mid-range parameters
###############################################################################

N_ITERATIONS = 10
RANDOM_SEED_BASE = 42

config = SimulationConfig(
    n_samples=500,
    n_states=3,
    n_informative=15,
    n_total_features=60,  # Mid-range feature dimension
    delta=0.5,  # Mid-range separation
    lambda_0=10.0,
    persistence=0.97,
    distribution_type="Poisson",
    correlated_noise=False,
    random_seed=42  # Will be updated in each iteration
)

# Mid-range hyperparameters (no grid search)
n_components = 3
jump_penalty = 10.0  # Middle of [0.1, 100]
max_feats = int(np.sqrt(config.n_total_features))**2  # √60 ≈ 7.75, so κ² ≈ 60

print(f"\nData Configuration:")
print(f"  - Samples: {config.n_samples}")
print(f"  - States: {config.n_states}")
print(f"  - Features: {config.n_total_features} ({config.n_informative} informative)")
print(f"  - Delta (separation): {config.delta}")
print(f"  - Distribution: {config.distribution_type}")

print(f"\nModel Hyperparameters:")
print(f"  - Number of states (K): {n_components}")
print(f"  - Jump penalty (λ): {jump_penalty}")
print(f"  - Max features (κ²): {max_feats}")

print(f"\nRunning {N_ITERATIONS} iterations...")
print(f"  - First iteration: Generate all visualizations")
print(f"  - All iterations: Collect BAC and Chamfer distance metrics")

# Storage for all iterations
all_results = {
    'Gaussian': {'BAC': [], 'Chamfer': [], 'Time': []},
    'Poisson': {'BAC': [], 'Chamfer': [], 'Time': []},
    'PoissonKL': {'BAC': [], 'Chamfer': [], 'Time': []}
}

###############################################################################
# Run Multiple Iterations
###############################################################################

for iteration in range(N_ITERATIONS):
    print(f"\n{'='*80}")
    print(f"Iteration {iteration + 1}/{N_ITERATIONS}")
    print(f"{'='*80}")
    
    # Update random seed for this iteration
    config.random_seed = RANDOM_SEED_BASE + iteration
    
    ###############################################################################
    # Generate Data
    ###############################################################################

    if iteration == 0:
        print(f"\nGenerating data...")
    X, states, breakpoints = generate_poisson_hmm_data(config)
    true_breakpoints = extract_breakpoints(states)

    if iteration == 0:
        print(f"  - Data shape: {X.shape}")
        print(f"  - True breakpoints: {len(true_breakpoints)} changes at indices {true_breakpoints}")

    ###############################################################################
    # Fit All Three Models
    ###############################################################################

    models = {}
    predictions = {}
    performance = {}

    model_specs = {
        'Gaussian': {'distribution': 'Gaussian', 'color': '#1f77b4'},
        'Poisson': {'distribution': 'Poisson', 'color': '#ff7f0e'},
        'PoissonKL': {'distribution': 'PoissonKL', 'color': '#2ca02c'}
    }

    if iteration == 0:
        print(f"\nFitting models...")
    for model_name, spec in model_specs.items():
        if iteration == 0:
            print(f"\n  Fitting {model_name}...")
        
        model = SparseJumpModel(
            n_components=n_components,
            max_feats=max_feats,
            jump_penalty=jump_penalty,
            distribution=spec['distribution'],
            max_iter=50,
            tol_w=1e-4,
            n_init_jm=10,
            verbose=0,
            random_state=config.random_seed
        )
        
        import time
        start_time = time.time()
        model.fit(X)
        fit_time = time.time() - start_time
        
        # Store model and predictions
        models[model_name] = model
        pred_states = model.labels_.values if hasattr(model.labels_, 'values') else model.labels_
        predictions[model_name] = pred_states
        
        # Compute metrics
        pred_breakpoints = extract_breakpoints(pred_states)
        bac = compute_bac_best_permutation(states, pred_states)
        chamfer = compute_chamfer_distance(true_breakpoints, pred_breakpoints)
        
        performance[model_name] = {
            'BAC': bac,
            'Chamfer': chamfer,
            'Time': fit_time,
            'Breakpoints': pred_breakpoints,
            'N_Breakpoints': len(pred_breakpoints)
        }
        
        # Store results for averaging
        all_results[model_name]['BAC'].append(bac)
        all_results[model_name]['Chamfer'].append(chamfer)
        all_results[model_name]['Time'].append(fit_time)
        
        if iteration == 0:
            print(f"    BAC: {bac:.4f}")
            print(f"    Chamfer Distance: {chamfer:.2f}")
            print(f"    Breakpoints detected: {len(pred_breakpoints)} (true: {len(true_breakpoints)})")
            print(f"    Computation time: {fit_time:.2f}s")
    
    # Only generate visualizations on first iteration
    if iteration > 0:
        continue

    ###############################################################################
    # Figure 1: Time Series with True and Predicted States
    ###############################################################################

print(f"\n\nCreating visualizations...")
print(f"1. Time series with states and breakpoints...")

# Select features to plot: top informative and top noise from each model
informative_features = [col for col in X.columns if 'informative' in col]
noise_features = [col for col in X.columns if 'noise' in col]

# Get top weighted features from each model
features_to_plot = set()
for model_name, model in models.items():
    top_informative = model.feat_weights[informative_features].nlargest(2).index.tolist()
    top_noise = model.feat_weights[noise_features].nlargest(1).index.tolist()
    features_to_plot.update(top_informative)
    features_to_plot.update(top_noise)

features_to_plot = sorted(list(features_to_plot), 
                         key=lambda x: (0 if 'informative' in x else 1, x))[:6]

print(f"   Plotting features: {features_to_plot}")

# Create figure with 4 columns: Data, Gaussian, Poisson, PoissonKL
n_features = len(features_to_plot)
fig, axes = plt.subplots(n_features, 4, figsize=(28, 4 * n_features), sharex=True)

if n_features == 1:
    axes = axes.reshape(1, -1)

fig.suptitle('Model Comparison on Poisson Data: True States vs Predictions', 
             fontsize=30, fontweight='bold', y=0.998)

colors_features = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e67e22']

# Column 0: True states
for i, feature in enumerate(features_to_plot):
    ax = axes[i, 0]
    
    # Plot time series
    ax.plot(X.index, X[feature], color=colors_features[i % len(colors_features)], 
            alpha=0.8, linewidth=2.5, zorder=2)
    
    # Add true breakpoints (green solid lines)
    for bp in true_breakpoints:
        ax.axvline(x=bp, color='#27ae60', linestyle='-', linewidth=3, alpha=0.6, zorder=3)
    
    # Formatting
    if i == 0:
        ax.set_title(f'True States', fontsize=26, fontweight='bold', pad=15)
    
    ax.set_ylabel(f'{feature}\nCount', fontsize=20, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=18)

# Columns 1-3: Model predictions
for j, model_name in enumerate(['Gaussian', 'Poisson', 'PoissonKL'], start=1):
    model = models[model_name]
    pred_breakpoints = performance[model_name]['Breakpoints']
    color = model_specs[model_name]['color']
    
    for i, feature in enumerate(features_to_plot):
        ax = axes[i, j]
        
        # Plot time series
        ax.plot(X.index, X[feature], color=colors_features[i % len(colors_features)], 
                alpha=0.8, linewidth=2.5, zorder=2)
        
        # Add true breakpoints (green, lighter)
        for bp in true_breakpoints:
            ax.axvline(x=bp, color='#27ae60', linestyle='-', linewidth=2, alpha=0.3, zorder=3)
        
        # Add predicted breakpoints (model-specific color)
        for bp in pred_breakpoints:
            ax.axvline(x=bp, color=color, linestyle='--', linewidth=2.5, alpha=0.8, zorder=4)
        
        # Get feature weight
        weight = model.feat_weights[feature]
        
        # Formatting
        if i == 0:
            bac = performance[model_name]['BAC']
            chamfer = performance[model_name]['Chamfer']
            ax.set_title(f'{model_name}\nBAC={bac:.3f}, Chamfer={chamfer:.1f}', 
                        fontsize=24, fontweight='bold', pad=15)
        
        # Add weight annotation
        ax.text(0.02, 0.98, f'w={weight:.3f}', 
                transform=ax.transAxes, fontsize=18,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.tick_params(labelsize=18)

# Add legend to top-left plot
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#27ae60', linestyle='-', linewidth=3, label='True Breakpoint'),
    Line2D([0], [0], color='gray', linestyle='--', linewidth=2.5, label='Predicted Breakpoint')
]
axes[0, 0].legend(handles=legend_elements, loc='upper left', fontsize=18, 
                 frameon=True, shadow=True)

# X-axis labels
for j in range(4):
    axes[-1, j].set_xlabel('Time Step', fontsize=22, fontweight='bold')

plt.tight_layout(rect=[0, 0.01, 1, 0.995])
plt.savefig(output_dir / "comparison_timeseries.jpg", 
            dpi=300, bbox_inches='tight', format='jpg')
plt.close()

print(f"   ✓ Saved: {output_dir / 'comparison_timeseries.jpg'}")

###############################################################################
# Figure 2: Feature Weights Comparison
###############################################################################

print(f"2. Feature weights comparison...")

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
    
    # Plot bars
    ax.bar(x_informative, informative_weights, color='#2ecc71', alpha=0.8, 
           edgecolor='black', linewidth=1.5, label='Informative')
    ax.bar(x_noise, noise_weights, color='#e74c3c', alpha=0.8, 
           edgecolor='black', linewidth=1.5, label='Noise')
    
    # Add horizontal line at mean weight
    mean_weight = weights.mean()
    ax.axhline(y=mean_weight, color='blue', linestyle='--', linewidth=2, 
               alpha=0.6, label=f'Mean: {mean_weight:.3f}')
    
    # Formatting
    bac = performance[model_name]['BAC']
    title_text = f'{model_name}'
    if model_name == 'PoissonKL':
        title_text += ' (sqrt weights)'
    title_text += f'\nBAC = {bac:.4f}'
    
    ax.set_title(title_text, fontsize=26, fontweight='bold', pad=15)
    ax.set_xlabel('Feature Index', fontsize=22, fontweight='bold')
    ax.set_ylabel('Feature Weight', fontsize=22, fontweight='bold')
    ax.legend(fontsize=18, frameon=True, shadow=True, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=18)
    
    # Add vertical separator
    if n_informative > 0 and n_noise > 0:
        ax.axvline(x=n_informative - 0.5, color='black', linestyle=':', 
                  linewidth=2, alpha=0.5)
        ax.text(n_informative/2, ax.get_ylim()[1]*0.95, 'Informative', 
               ha='center', fontsize=16, fontweight='bold')
        ax.text(n_informative + n_noise/2, ax.get_ylim()[1]*0.95, 'Noise', 
               ha='center', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "comparison_feature_weights.jpg", 
            dpi=300, bbox_inches='tight', format='jpg')
plt.close()

print(f"   ✓ Saved: {output_dir / 'comparison_feature_weights.jpg'}")

###############################################################################
# Figure 3: Performance Metrics Comparison (will be generated after all iterations)
###############################################################################

# This figure will be generated after all iterations with mean/std values
# Placeholder - actual generation happens after the iteration loop

###############################################################################
# Figure 4: Breakpoint Alignment
###############################################################################

print(f"4. Breakpoint alignment comparison...")

fig, ax = plt.subplots(1, 1, figsize=(24, 8))

# Plot true breakpoints at y=0
ax.scatter(true_breakpoints, [0]*len(true_breakpoints), 
           s=400, marker='o', color='#27ae60', alpha=0.9, 
           label=f'True ({len(true_breakpoints)} breakpoints)', 
           zorder=5, edgecolor='black', linewidth=2.5)

# Plot predicted breakpoints for each model at different y-levels
y_positions = {'Gaussian': 1, 'Poisson': 2, 'PoissonKL': 3}
markers = {'Gaussian': 's', 'Poisson': '^', 'PoissonKL': 'D'}

for model_name in ['Gaussian', 'Poisson', 'PoissonKL']:
    breakpts = performance[model_name]['Breakpoints']
    y_pos = y_positions[model_name]
    color = model_specs[model_name]['color']
    bac = performance[model_name]['BAC']
    chamfer = performance[model_name]['Chamfer']
    
    ax.scatter(breakpts, [y_pos]*len(breakpts), 
               s=400, marker=markers[model_name], 
               color=color, alpha=0.9,
               label=f'{model_name} ({len(breakpts)} bp, BAC={bac:.3f}, Ch={chamfer:.1f})', 
               zorder=5, edgecolor='black', linewidth=2.5)
    
    # Draw vertical lines from predicted to true for alignment visualization
    for bp in breakpts:
        ax.plot([bp, bp], [0, y_pos], color='gray', alpha=0.15, 
               linewidth=1.5, zorder=1)

ax.set_xlabel('Time Step', fontsize=24, fontweight='bold')
ax.set_ylabel('', fontsize=22)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(['True', 'Gaussian', 'Poisson', 'PoissonKL'], 
                   fontsize=22, fontweight='bold')
ax.set_title('Breakpoint Alignment: True vs Predicted', 
            fontsize=28, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=20, frameon=True, shadow=True, ncol=1)
ax.grid(True, axis='x', linestyle='--', alpha=0.4)
ax.set_ylim(-0.5, 3.5)
ax.tick_params(labelsize=20)

plt.tight_layout()
plt.savefig(output_dir / "comparison_breakpoint_alignment.jpg", 
            dpi=300, bbox_inches='tight', format='jpg')
plt.close()

print(f"   ✓ Saved: {output_dir / 'comparison_breakpoint_alignment.jpg'}")

###############################################################################
# Figure 5: Stacked States Plot (like sparse_jump_test.py)
###############################################################################

print(f"5. Stacked states comparison...")

# Select best informative feature (highest average weight across models)
avg_weights = {}
for feature in informative_features:
    avg_weight = np.mean([models[m].feat_weights[feature] for m in ['Gaussian', 'Poisson', 'PoissonKL']])
    avg_weights[feature] = avg_weight

best_informative = max(avg_weights.items(), key=lambda x: x[1])[0]

# Create stacked plot
n_models = 3  # Gaussian, Poisson, PoissonKL
n_panels = 2 + n_models  # Data + True States + 3 models

fig, axes = plt.subplots(n_panels, 1, figsize=(22, 3 * n_panels), sharex=True)

fig.suptitle(f'Stacked States Comparison: {best_informative}', 
             fontsize=28, fontweight='bold', y=0.995)

panel_idx = 0

# Panel 0: Time series data
ax = axes[panel_idx]
ax.plot(X.index, X[best_informative], color='#2c3e50', linewidth=2.5, alpha=0.8)
ax.set_ylabel('Count', fontsize=20, fontweight='bold')
ax.set_title(f'{best_informative}', fontsize=24, fontweight='bold', pad=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(axis='y', labelsize=18)
panel_idx += 1

# Define color map for states
n_states = len(np.unique(states))
cmap = plt.colormaps.get_cmap('Set2')
color_map = {i: cmap(i / max(n_states - 1, 1)) for i in range(n_states)}

def plot_state_bars(ax, labels, label_text, bac=None):
    """Plot horizontal state bars."""
    labels_array = np.asarray(labels)
    
    # Plot horizontal bars for each state
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
    
    # Title on the left, outside the plot
    ax.set_ylabel('State', fontsize=18, fontweight='bold')
    ax.set_title(label_text, fontsize=22, fontweight='bold', pad=10, loc='left')
    
    # Add BAC as text annotation to the right of the plot
    if bac is not None:
        ax.text(1.02, 0.5, f'BAC={bac:.3f}', 
                transform=ax.transAxes, fontsize=20, fontweight='bold',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=2))
    
    ax.set_ylim(0.5, n_states + 0.5)
    ax.set_yticks(range(1, n_states + 1))
    ax.set_yticklabels([f'{i}' for i in range(n_states)], fontsize=18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=18)
    ax.grid(True, alpha=0.2, axis='x', linestyle='--')

# Panel 1: True states
ax = axes[panel_idx]
plot_state_bars(ax, states, 'True States')
panel_idx += 1

# Panels 2-4: Model predictions
for model_name in ['Gaussian', 'Poisson', 'PoissonKL']:
    ax = axes[panel_idx]
    pred_states = predictions[model_name]
    bac = performance[model_name]['BAC']
    plot_state_bars(ax, pred_states, f'{model_name}', bac=bac)
    panel_idx += 1

# X-axis label on bottom panel
axes[-1].set_xlabel('Time Step', fontsize=22, fontweight='bold')
axes[-1].tick_params(labelsize=18)

plt.tight_layout(rect=[0, 0.01, 1, 0.995])
plt.subplots_adjust(hspace=0.25)
plt.savefig(output_dir / "comparison_stacked_states.jpg", 
            dpi=300, bbox_inches='tight', format='jpg')
plt.close()

print(f"   ✓ Saved: {output_dir / 'comparison_stacked_states.jpg'}")

###############################################################################
# Compute and Print Aggregated Results
###############################################################################

print(f"\n" + "="*80)
print(f"All {N_ITERATIONS} Iterations Complete!")
print("="*80)

###############################################################################
# Generate Figure 3: Performance Metrics with Error Bars
###############################################################################

print(f"\nGenerating aggregated performance metrics plot...")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

metrics = ['BAC', 'Chamfer', 'Time']
metric_labels = {
    'BAC': 'Balanced Accuracy',
    'Chamfer': 'Chamfer Distance',
    'Time': 'Time (s)'
}
titles = ['Balanced Accuracy\n(Higher is Better)', 
          'Chamfer Distance\n(Lower is Better)', 
          'Computation Time (seconds)\n(Lower is Better)']
colors_bars = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx]
    
    # Compute mean and std for each model
    means = []
    stds = []
    for model_name in ['Gaussian', 'Poisson', 'PoissonKL']:
        values = all_results[model_name][metric]
        means.append(np.mean(values))
        stds.append(np.std(values))
    
    x_pos = np.arange(3)
    
    # Plot bars with error bars
    bars = ax.bar(x_pos, means, yerr=stds, color=colors_bars, alpha=0.8, 
                  edgecolor='black', linewidth=2, capsize=10, 
                  error_kw={'linewidth': 3, 'ecolor': 'black', 'alpha': 0.7})
    
    # Add value labels on bars (mean ± std)
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        if metric == 'Time':
            label_text = f'{mean_val:.2f}±{std_val:.2f}s'
        elif metric == 'BAC':
            label_text = f'{mean_val:.4f}±{std_val:.4f}'
        else:
            label_text = f'{mean_val:.2f}±{std_val:.2f}'
        
        ax.text(bar.get_x() + bar.get_width()/2., height + std_val,
                label_text,
                ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    ax.set_ylabel(metric_labels[metric], fontsize=22, fontweight='bold')
    ax.set_title(title, fontsize=24, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Gaussian', 'Poisson', 'PoissonKL'], 
                       fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=18)
    
    # Add subtitle with iteration count
    ax.text(0.5, 0.98, f'Mean ± Std (N={N_ITERATIONS})', 
            transform=ax.transAxes, ha='center', va='top',
            fontsize=16, style='italic', color='gray')

plt.tight_layout()
plt.savefig(output_dir / "comparison_metrics.jpg", 
            dpi=300, bbox_inches='tight', format='jpg')
plt.close()

print(f"   ✓ Saved: {output_dir / 'comparison_metrics.jpg'}")

# Compute means and standard deviations
print(f"\nAggregated Results Across {N_ITERATIONS} Iterations:")
print(f"{'='*80}")

print(f"\nBalanced Accuracy (BAC):")
print(f"{'Model':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print(f"{'-'*54}")
for model_name in ['Gaussian', 'Poisson', 'PoissonKL']:
    bac_values = all_results[model_name]['BAC']
    mean_bac = np.mean(bac_values)
    std_bac = np.std(bac_values)
    min_bac = np.min(bac_values)
    max_bac = np.max(bac_values)
    print(f"{model_name:<12} {mean_bac:>10.4f} {std_bac:>10.4f} {min_bac:>10.4f} {max_bac:>10.4f}")

print(f"\nChamfer Distance:")
print(f"{'Model':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print(f"{'-'*54}")
for model_name in ['Gaussian', 'Poisson', 'PoissonKL']:
    chamfer_values = all_results[model_name]['Chamfer']
    mean_chamfer = np.mean(chamfer_values)
    std_chamfer = np.std(chamfer_values)
    min_chamfer = np.min(chamfer_values)
    max_chamfer = np.max(chamfer_values)
    print(f"{model_name:<12} {mean_chamfer:>10.2f} {std_chamfer:>10.2f} {min_chamfer:>10.2f} {max_chamfer:>10.2f}")

print(f"\nComputation Time (seconds):")
print(f"{'Model':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print(f"{'-'*54}")
for model_name in ['Gaussian', 'Poisson', 'PoissonKL']:
    time_values = all_results[model_name]['Time']
    mean_time = np.mean(time_values)
    std_time = np.std(time_values)
    min_time = np.min(time_values)
    max_time = np.max(time_values)
    print(f"{model_name:<12} {mean_time:>10.2f} {std_time:>10.2f} {min_time:>10.2f} {max_time:>10.2f}")

###############################################################################
# Print Summary from First Iteration
###############################################################################

print(f"\n" + "="*80)
print("First Iteration Summary (Visualizations Generated)")
print("="*80)

print(f"\nConfiguration Summary:")
print(f"  N = {config.n_samples} samples")
print(f"  K = {config.n_states} states")
print(f"  P = {config.n_total_features} features ({config.n_informative} informative)")
print(f"  δ = {config.delta} (state separation)")
print(f"  Hyperparameters: K={n_components}, λ={jump_penalty}, κ²={max_feats}")

print(f"\nTrue Data:")
print(f"  Breakpoints: {len(true_breakpoints)}")
print(f"  Locations: {true_breakpoints}")

print(f"\nPerformance Summary:")
print(f"{'Model':<12} {'BAC':>8} {'Chamfer':>10} {'Time (s)':>10} {'N_BP':>6}")
print(f"{'-'*50}")
for model_name in ['Gaussian', 'Poisson', 'PoissonKL']:
    perf = performance[model_name]
    print(f"{model_name:<12} {perf['BAC']:>8.4f} {perf['Chamfer']:>10.2f} "
          f"{perf['Time']:>10.2f} {perf['N_Breakpoints']:>6}")

print(f"\nFeature Selection Quality:")
print(f"{'Model':<12} {'Mean Informative':>18} {'Mean Noise':>12} {'Ratio':>8}")
print(f"{'-'*54}")
for model_name in ['Gaussian', 'Poisson', 'PoissonKL']:
    model = models[model_name]
    mean_inf = model.feat_weights[informative_features].mean()
    mean_noise = model.feat_weights[noise_features].mean()
    ratio = mean_inf / mean_noise if mean_noise > 0 else np.inf
    print(f"{model_name:<12} {mean_inf:>18.4f} {mean_noise:>12.4f} {ratio:>8.2f}")

# For PoissonKL, also show sqrt(w) for comparison with Gaussian/Poisson
print(f"\nFeature Selection Quality (sqrt weights for PoissonKL):")
print(f"{'Model':<12} {'Mean Informative':>18} {'Mean Noise':>12} {'Ratio':>8}")
print(f"{'-'*54}")
for model_name in ['Gaussian', 'Poisson', 'PoissonKL']:
    model = models[model_name]
    
    # For PoissonKL, compute sqrt of feat_weights to compare with Gaussian/Poisson
    if model_name == 'PoissonKL':
        # feat_weights = w for PoissonKL, so sqrt(feat_weights) = sqrt(w)
        sqrt_weights = np.sqrt(model.feat_weights)
        mean_inf = sqrt_weights[informative_features].mean()
        mean_noise = sqrt_weights[noise_features].mean()
    else:
        # For Gaussian/Poisson, feat_weights already = sqrt(w)
        mean_inf = model.feat_weights[informative_features].mean()
        mean_noise = model.feat_weights[noise_features].mean()
    
    ratio = mean_inf / mean_noise if mean_noise > 0 else np.inf
    
    suffix = " (sqrt applied)" if model_name == 'PoissonKL' else ""
    print(f"{model_name:<12} {mean_inf:>18.4f} {mean_noise:>12.4f} {ratio:>8.2f}{suffix}")

print(f"\nGenerated figures:")
print(f"  1. comparison_timeseries.jpg - Time series with all models (1st iteration)")
print(f"  2. comparison_feature_weights.jpg - Feature weight comparison (1st iteration)")
print(f"  3. comparison_metrics.jpg - Performance metrics with error bars (Mean ± Std, N={N_ITERATIONS})")
print(f"  4. comparison_breakpoint_alignment.jpg - Breakpoint alignment (1st iteration)")
print(f"  5. comparison_stacked_states.jpg - Stacked states visualization (1st iteration)")

print(f"\nAll figures saved to: {output_dir}/")
print(f"\nBest model by BAC: {max(performance.items(), key=lambda x: x[1]['BAC'])[0]}")
print(f"Best model by Chamfer: {min(performance.items(), key=lambda x: x[1]['Chamfer'])[0]}")
