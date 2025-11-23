"""
Visualize Results from Aggregated CSV Files

This script loads the aggregated results from simulation_runner_parallel.py
and creates publication-quality plots showing model comparison.

Usage:
    python visualize_aggregated_results.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality figures
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 22,
    'figure.dpi': 300,
})

print("="*80)
print("Visualizing Simulation Results from CSV")
print("="*80)

# Load results
results_file = Path("results/aggregated/all_results.csv")
if not results_file.exists():
    print(f"\nError: Results file not found: {results_file}")
    print("Run the simulation first:")
    print("  python simulation_runner_parallel.py --quick_test")
    exit(1)

df = pd.read_csv(results_file)
print(f"\nLoaded {len(df)} results")
print(f"Models: {df['model_name'].unique()}")
print(f"Configurations: {len(df.groupby(['delta', 'n_total_features']))}")

# Create output directory
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

###############################################################################
# Figure 1: Model Comparison - Balanced Accuracy by Delta
###############################################################################
print("\n1. Creating BAC by Delta comparison...")

fig, ax = plt.subplots(figsize=(14, 8))

colors = {'Gaussian': '#1f77b4', 'Poisson': '#ff7f0e', 'PoissonKL': '#2ca02c'}
markers = {'Gaussian': 'o', 'Poisson': 's', 'PoissonKL': '^'}

for model in ['Gaussian', 'Poisson', 'PoissonKL']:
    model_data = df[df['model_name'] == model]
    delta_stats = model_data.groupby('delta')['balanced_accuracy'].agg(['mean', 'std', 'count'])
    
    # Plot with error bars (if multiple replications)
    if delta_stats['count'].min() > 1:
        ax.errorbar(delta_stats.index, delta_stats['mean'], 
                   yerr=delta_stats['std'],
                   marker=markers[model], markersize=12, linewidth=3,
                   capsize=8, capthick=2, label=model, color=colors[model])
    else:
        ax.plot(delta_stats.index, delta_stats['mean'],
               marker=markers[model], markersize=12, linewidth=3,
               label=model, color=colors[model])

ax.set_xlabel('Delta (State Separation)', fontsize=20, fontweight='bold')
ax.set_ylabel('Balanced Accuracy', fontsize=20, fontweight='bold')
ax.set_title('Model Performance vs. State Separation\n(Higher δ = Better Separation)', 
             fontsize=22, fontweight='bold', pad=20)
ax.legend(fontsize=18, frameon=True, shadow=True, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_ylim(0.9, 1.02)

plt.tight_layout()
plt.savefig(output_dir / "01_bac_by_delta.jpg", dpi=300, bbox_inches='tight', format='jpg')
plt.close()
print(f"   ✓ Saved: {output_dir / '01_bac_by_delta.jpg'}")

###############################################################################
# Figure 2: Model Comparison - Chamfer Distance by Delta
###############################################################################
print("2. Creating Chamfer Distance by Delta...")

fig, ax = plt.subplots(figsize=(14, 8))

for model in ['Gaussian', 'Poisson', 'PoissonKL']:
    model_data = df[df['model_name'] == model]
    delta_stats = model_data.groupby('delta')['chamfer_distance'].agg(['mean', 'std', 'count'])
    
    if delta_stats['count'].min() > 1:
        ax.errorbar(delta_stats.index, delta_stats['mean'], 
                   yerr=delta_stats['std'],
                   marker=markers[model], markersize=12, linewidth=3,
                   capsize=8, capthick=2, label=model, color=colors[model])
    else:
        ax.plot(delta_stats.index, delta_stats['mean'],
               marker=markers[model], markersize=12, linewidth=3,
               label=model, color=colors[model])

ax.set_xlabel('Delta (State Separation)', fontsize=20, fontweight='bold')
ax.set_ylabel('Chamfer Distance', fontsize=20, fontweight='bold')
ax.set_title('Breakpoint Alignment vs. State Separation\n(Lower is Better)', 
             fontsize=22, fontweight='bold', pad=20)
ax.legend(fontsize=18, frameon=True, shadow=True, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "02_chamfer_by_delta.jpg", dpi=300, bbox_inches='tight', format='jpg')
plt.close()
print(f"   ✓ Saved: {output_dir / '02_chamfer_by_delta.jpg'}")

###############################################################################
# Figure 3: Performance by Feature Dimension
###############################################################################
print("3. Creating performance by feature dimension...")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# BAC by P
for model in ['Gaussian', 'Poisson', 'PoissonKL']:
    model_data = df[df['model_name'] == model]
    p_stats = model_data.groupby('n_total_features')['balanced_accuracy'].agg(['mean', 'std', 'count'])
    
    if p_stats['count'].min() > 1:
        axes[0].errorbar(p_stats.index, p_stats['mean'], 
                        yerr=p_stats['std'],
                        marker=markers[model], markersize=12, linewidth=3,
                        capsize=8, capthick=2, label=model, color=colors[model])
    else:
        axes[0].plot(p_stats.index, p_stats['mean'],
                    marker=markers[model], markersize=12, linewidth=3,
                    label=model, color=colors[model])

axes[0].set_xlabel('Number of Features (P)', fontsize=18, fontweight='bold')
axes[0].set_ylabel('Balanced Accuracy', fontsize=18, fontweight='bold')
axes[0].set_title('BAC vs. Feature Dimension', fontsize=20, fontweight='bold', pad=15)
axes[0].legend(fontsize=16, frameon=True, shadow=True)
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale('log')

# Chamfer by P
for model in ['Gaussian', 'Poisson', 'PoissonKL']:
    model_data = df[df['model_name'] == model]
    p_stats = model_data.groupby('n_total_features')['chamfer_distance'].agg(['mean', 'std', 'count'])
    
    if p_stats['count'].min() > 1:
        axes[1].errorbar(p_stats.index, p_stats['mean'], 
                        yerr=p_stats['std'],
                        marker=markers[model], markersize=12, linewidth=3,
                        capsize=8, capthick=2, label=model, color=colors[model])
    else:
        axes[1].plot(p_stats.index, p_stats['mean'],
                    marker=markers[model], markersize=12, linewidth=3,
                    label=model, color=colors[model])

axes[1].set_xlabel('Number of Features (P)', fontsize=18, fontweight='bold')
axes[1].set_ylabel('Chamfer Distance', fontsize=18, fontweight='bold')
axes[1].set_title('Chamfer Distance vs. Feature Dimension', fontsize=20, fontweight='bold', pad=15)
axes[1].legend(fontsize=16, frameon=True, shadow=True)
axes[1].grid(True, alpha=0.3)
axes[1].set_xscale('log')

plt.tight_layout()
plt.savefig(output_dir / "03_performance_by_features.jpg", dpi=300, bbox_inches='tight', format='jpg')
plt.close()
print(f"   ✓ Saved: {output_dir / '03_performance_by_features.jpg'}")

###############################################################################
# Figure 4: Overall Model Comparison
###############################################################################
print("4. Creating overall model comparison...")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# BAC
model_stats_bac = df.groupby('model_name')['balanced_accuracy'].agg(['mean', 'std', 'count'])
x_pos = np.arange(len(model_stats_bac))
colors_bars = ['#1f77b4', '#ff7f0e', '#2ca02c']

if model_stats_bac['count'].min() > 1:
    axes[0].bar(x_pos, model_stats_bac['mean'], 
               yerr=model_stats_bac['std'],
               capsize=10, color=colors_bars, edgecolor='black', 
               linewidth=2, alpha=0.8)
else:
    axes[0].bar(x_pos, model_stats_bac['mean'],
               color=colors_bars, edgecolor='black', 
               linewidth=2, alpha=0.8)

axes[0].set_ylabel('Balanced Accuracy', fontsize=20, fontweight='bold')
axes[0].set_title('Balanced Accuracy\n(Higher is Better)', fontsize=22, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(model_stats_bac.index, fontsize=18, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Chamfer Distance
model_stats_chamfer = df.groupby('model_name')['chamfer_distance'].agg(['mean', 'std', 'count'])

if model_stats_chamfer['count'].min() > 1:
    axes[1].bar(x_pos, model_stats_chamfer['mean'], 
               yerr=model_stats_chamfer['std'],
               capsize=10, color=colors_bars, edgecolor='black', 
               linewidth=2, alpha=0.8)
else:
    axes[1].bar(x_pos, model_stats_chamfer['mean'],
               color=colors_bars, edgecolor='black', 
               linewidth=2, alpha=0.8)

axes[1].set_ylabel('Chamfer Distance', fontsize=20, fontweight='bold')
axes[1].set_title('Chamfer Distance\n(Lower is Better)', fontsize=22, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(model_stats_chamfer.index, fontsize=18, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Computation Time
model_stats_time = df.groupby('model_name')['computation_time'].agg(['mean', 'std', 'count'])

if model_stats_time['count'].min() > 1:
    axes[2].bar(x_pos, model_stats_time['mean'], 
               yerr=model_stats_time['std'],
               capsize=10, color=colors_bars, edgecolor='black', 
               linewidth=2, alpha=0.8)
else:
    axes[2].bar(x_pos, model_stats_time['mean'],
               color=colors_bars, edgecolor='black', 
               linewidth=2, alpha=0.8)

axes[2].set_ylabel('Computation Time (seconds)', fontsize=20, fontweight='bold')
axes[2].set_title('Computation Time\n(Lower is Better)', fontsize=22, fontweight='bold')
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(model_stats_time.index, fontsize=18, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "04_overall_comparison.jpg", dpi=300, bbox_inches='tight', format='jpg')
plt.close()
print(f"   ✓ Saved: {output_dir / '04_overall_comparison.jpg'}")

###############################################################################
# Figure 5: Heatmap - BAC by Delta and P
###############################################################################
print("5. Creating heatmap of BAC by delta and P...")

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

for idx, model in enumerate(['Gaussian', 'Poisson', 'PoissonKL']):
    model_data = df[df['model_name'] == model]
    
    # Create pivot table
    pivot = model_data.pivot_table(
        values='balanced_accuracy',
        index='delta',
        columns='n_total_features',
        aggfunc='mean'
    )
    
    # Plot heatmap
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0.95, vmax=1.0, center=0.975,
                cbar_kws={'label': 'BAC'},
                ax=axes[idx], linewidths=1, linecolor='gray')
    
    axes[idx].set_title(f'{model}', fontsize=22, fontweight='bold', pad=15)
    axes[idx].set_xlabel('Number of Features (P)', fontsize=18, fontweight='bold')
    axes[idx].set_ylabel('Delta (δ)', fontsize=18, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "05_bac_heatmap.jpg", dpi=300, bbox_inches='tight', format='jpg')
plt.close()
print(f"   ✓ Saved: {output_dir / '05_bac_heatmap.jpg'}")

###############################################################################
# Figure 6: Feature Selection Quality
###############################################################################
print("6. Creating feature selection comparison...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# F1 Score
for model in ['Gaussian', 'Poisson', 'PoissonKL']:
    model_data = df[df['model_name'] == model]
    delta_stats = model_data.groupby('delta')['feature_f1'].agg(['mean', 'std', 'count'])
    
    if delta_stats['count'].min() > 1:
        axes[0].errorbar(delta_stats.index, delta_stats['mean'], 
                        yerr=delta_stats['std'],
                        marker=markers[model], markersize=12, linewidth=3,
                        capsize=8, capthick=2, label=model, color=colors[model])
    else:
        axes[0].plot(delta_stats.index, delta_stats['mean'],
                    marker=markers[model], markersize=12, linewidth=3,
                    label=model, color=colors[model])

axes[0].set_xlabel('Delta (State Separation)', fontsize=18, fontweight='bold')
axes[0].set_ylabel('Feature Selection F1', fontsize=18, fontweight='bold')
axes[0].set_title('Feature Selection Quality', fontsize=20, fontweight='bold', pad=15)
axes[0].legend(fontsize=16, frameon=True, shadow=True)
axes[0].grid(True, alpha=0.3)

# Noise Features Selected
for model in ['Gaussian', 'Poisson', 'PoissonKL']:
    model_data = df[df['model_name'] == model]
    delta_stats = model_data.groupby('delta')['n_selected_noise'].agg(['mean', 'std', 'count'])
    
    if delta_stats['count'].min() > 1:
        axes[1].errorbar(delta_stats.index, delta_stats['mean'], 
                        yerr=delta_stats['std'],
                        marker=markers[model], markersize=12, linewidth=3,
                        capsize=8, capthick=2, label=model, color=colors[model])
    else:
        axes[1].plot(delta_stats.index, delta_stats['mean'],
                    marker=markers[model], markersize=12, linewidth=3,
                    label=model, color=colors[model])

axes[1].set_xlabel('Delta (State Separation)', fontsize=18, fontweight='bold')
axes[1].set_ylabel('Noise Features Selected', fontsize=18, fontweight='bold')
axes[1].set_title('Noise Features Selected\n(Lower is Better)', fontsize=20, fontweight='bold', pad=15)
axes[1].legend(fontsize=16, frameon=True, shadow=True)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "06_feature_selection.jpg", dpi=300, bbox_inches='tight', format='jpg')
plt.close()
print(f"   ✓ Saved: {output_dir / '06_feature_selection.jpg'}")

###############################################################################
# Print Summary Statistics
###############################################################################
print("\n" + "="*80)
print("Summary Statistics")
print("="*80)

print("\n1. Overall Performance by Model:")
print("-" * 80)
summary = df.groupby('model_name').agg({
    'balanced_accuracy': ['mean', 'std', 'min', 'max'],
    'chamfer_distance': ['mean', 'std', 'min', 'max'],
    'feature_f1': ['mean', 'std', 'min', 'max'],
    'computation_time': ['mean', 'std', 'min', 'max']
})
print(summary)

print("\n2. Performance by Delta (Separation):")
print("-" * 80)
for delta in sorted(df['delta'].unique()):
    print(f"\nDelta = {delta}:")
    delta_data = df[df['delta'] == delta]
    for model in ['Gaussian', 'Poisson', 'PoissonKL']:
        model_data = delta_data[delta_data['model_name'] == model]
        bac_mean = model_data['balanced_accuracy'].mean()
        chamfer_mean = model_data['chamfer_distance'].mean()
        print(f"  {model:10s}: BAC = {bac_mean:.4f}, Chamfer = {chamfer_mean:.4f}")

print("\n" + "="*80)
print("Visualization Complete!")
print("="*80)
print(f"\nAll figures saved to: {output_dir}/")
print("\nGenerated figures:")
print("  1. 01_bac_by_delta.jpg - BAC vs separation level")
print("  2. 02_chamfer_by_delta.jpg - Chamfer distance vs separation")
print("  3. 03_performance_by_features.jpg - Performance vs feature dimension")
print("  4. 04_overall_comparison.jpg - Overall model comparison")
print("  5. 05_bac_heatmap.jpg - BAC heatmap (delta × P)")
print("  6. 06_feature_selection.jpg - Feature selection quality")
print("\nAll figures are:")
print("  - High resolution (300 DPI)")
print("  - JPEG format for report inclusion")
print("  - Publication quality")
