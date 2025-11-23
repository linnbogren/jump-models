"""
Visualization script for simulation results.
Creates publication-quality figures with large text for reports.
"""

import pandas as pd
import numpy as np
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
    'figure.dpi': 300,  # High DPI for JPEG
})

# Create output directory
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Load results
results_df = pd.read_csv("results/aggregated/all_results.csv")
grid_df = pd.read_csv("results/grid_search/all_grid_results.csv")

print("="*80)
print("Visualizing Simulation Results")
print("="*80)
print(f"\nLoaded {len(results_df)} best model results")
print(f"Loaded {len(grid_df)} grid search evaluations")
print(f"\nModels: {results_df['model_name'].unique()}")
print(f"Configurations: {len(results_df.groupby(['delta', 'n_total_features']))}")

##############################################################################
# Figure 1: Model Comparison - Balanced Accuracy
##############################################################################
print("\n1. Creating Model Comparison - Balanced Accuracy...")

fig, ax = plt.subplots(figsize=(12, 8))

# Group by model and compute statistics
model_stats = results_df.groupby('model_name')['balanced_accuracy'].agg(['mean', 'std', 'count'])

# Create bar plot
x_pos = np.arange(len(model_stats))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

bars = ax.bar(x_pos, model_stats['mean'], 
               yerr=model_stats['std'], 
               capsize=10,
               color=colors,
               edgecolor='black',
               linewidth=2,
               alpha=0.8)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, model_stats['mean'], model_stats['std'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
            f'{mean:.3f}',
            ha='center', va='bottom', fontsize=18, fontweight='bold')

ax.set_xlabel('Model Type', fontsize=20, fontweight='bold')
ax.set_ylabel('Balanced Accuracy', fontsize=20, fontweight='bold')
ax.set_title('Model Comparison: Balanced Accuracy\n(Higher is Better)', 
             fontsize=22, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(model_stats.index, fontsize=18, fontweight='bold')
ax.set_ylim(0, max(model_stats['mean'] + model_stats['std']) * 1.15)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "01_model_comparison_bac.jpg", 
            dpi=300, bbox_inches='tight', format='jpg')
plt.close()
print(f"   ✓ Saved: {output_dir / '01_model_comparison_bac.jpg'}")

##############################################################################
# Figure 2: Model Comparison - Feature Selection F1
##############################################################################
print("\n2. Creating Model Comparison - Feature Selection F1...")

fig, ax = plt.subplots(figsize=(12, 8))

# Group by model and compute statistics
f1_stats = results_df.groupby('model_name')['feature_f1'].agg(['mean', 'std', 'count'])

# Create bar plot
bars = ax.bar(x_pos, f1_stats['mean'], 
               yerr=f1_stats['std'], 
               capsize=10,
               color=colors,
               edgecolor='black',
               linewidth=2,
               alpha=0.8)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, f1_stats['mean'], f1_stats['std'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
            f'{mean:.3f}',
            ha='center', va='bottom', fontsize=18, fontweight='bold')

ax.set_xlabel('Model Type', fontsize=20, fontweight='bold')
ax.set_ylabel('Feature Selection F1 Score', fontsize=20, fontweight='bold')
ax.set_title('Model Comparison: Feature Selection Quality\n(Higher is Better)', 
             fontsize=22, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(f1_stats.index, fontsize=18, fontweight='bold')
ax.set_ylim(0, max(f1_stats['mean'] + f1_stats['std']) * 1.15)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "02_model_comparison_f1.jpg", 
            dpi=300, bbox_inches='tight', format='jpg')
plt.close()
print(f"   ✓ Saved: {output_dir / '02_model_comparison_f1.jpg'}")

##############################################################################
# Figure 3: Performance by Delta (Separation Level)
##############################################################################
print("\n3. Creating Performance by Delta...")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Balanced Accuracy by Delta
for model in results_df['model_name'].unique():
    model_data = results_df[results_df['model_name'] == model]
    delta_stats = model_data.groupby('delta')['balanced_accuracy'].agg(['mean', 'std'])
    
    axes[0].errorbar(delta_stats.index, delta_stats['mean'], 
                     yerr=delta_stats['std'], 
                     marker='o', markersize=12, linewidth=3,
                     capsize=8, capthick=2, label=model)

axes[0].set_xlabel('Delta (Separation Level)', fontsize=20, fontweight='bold')
axes[0].set_ylabel('Balanced Accuracy', fontsize=20, fontweight='bold')
axes[0].set_title('Balanced Accuracy vs. Separation\n(Higher δ = Better Separation)', 
                  fontsize=22, fontweight='bold', pad=20)
axes[0].legend(fontsize=16, frameon=True, shadow=True, loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(labelsize=16)

# Feature F1 by Delta
for model in results_df['model_name'].unique():
    model_data = results_df[results_df['model_name'] == model]
    delta_stats = model_data.groupby('delta')['feature_f1'].agg(['mean', 'std'])
    
    axes[1].errorbar(delta_stats.index, delta_stats['mean'], 
                     yerr=delta_stats['std'], 
                     marker='s', markersize=12, linewidth=3,
                     capsize=8, capthick=2, label=model)

axes[1].set_xlabel('Delta (Separation Level)', fontsize=20, fontweight='bold')
axes[1].set_ylabel('Feature Selection F1', fontsize=20, fontweight='bold')
axes[1].set_title('Feature Selection vs. Separation\n(Higher δ = Better Separation)', 
                  fontsize=22, fontweight='bold', pad=20)
axes[1].legend(fontsize=16, frameon=True, shadow=True, loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelsize=16)

plt.tight_layout()
plt.savefig(output_dir / "03_performance_by_delta.jpg", 
            dpi=300, bbox_inches='tight', format='jpg')
plt.close()
print(f"   ✓ Saved: {output_dir / '03_performance_by_delta.jpg'}")

##############################################################################
# Figure 4: Computation Time Comparison
##############################################################################
print("\n4. Creating Computation Time Comparison...")

fig, ax = plt.subplots(figsize=(12, 8))

# Group by model and compute statistics
time_stats = results_df.groupby('model_name')['computation_time'].agg(['mean', 'std', 'count'])

# Create bar plot
bars = ax.bar(x_pos, time_stats['mean'], 
               yerr=time_stats['std'], 
               capsize=10,
               color=colors,
               edgecolor='black',
               linewidth=2,
               alpha=0.8)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, time_stats['mean'], time_stats['std'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
            f'{mean:.2f}s',
            ha='center', va='bottom', fontsize=18, fontweight='bold')

ax.set_xlabel('Model Type', fontsize=20, fontweight='bold')
ax.set_ylabel('Computation Time (seconds)', fontsize=20, fontweight='bold')
ax.set_title('Computational Efficiency Comparison\n(Lower is Better)', 
             fontsize=22, fontweight='bold', pad=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(time_stats.index, fontsize=18, fontweight='bold')
ax.set_ylim(0, max(time_stats['mean'] + time_stats['std']) * 1.15)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "04_computation_time.jpg", 
            dpi=300, bbox_inches='tight', format='jpg')
plt.close()
print(f"   ✓ Saved: {output_dir / '04_computation_time.jpg'}")

##############################################################################
# Figure 5: Grid Search - Hyperparameter Impact
##############################################################################
print("\n5. Creating Hyperparameter Impact Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Impact of n_components
for model in grid_df['model_name'].unique():
    model_data = grid_df[grid_df['model_name'] == model]
    comp_stats = model_data.groupby('n_components')['balanced_accuracy'].agg(['mean', 'std'])
    
    axes[0].errorbar(comp_stats.index, comp_stats['mean'], 
                     yerr=comp_stats['std'], 
                     marker='o', markersize=10, linewidth=3,
                     capsize=6, capthick=2, label=model)

axes[0].set_xlabel('Number of States (K)', fontsize=18, fontweight='bold')
axes[0].set_ylabel('Balanced Accuracy', fontsize=18, fontweight='bold')
axes[0].set_title('Impact of Number of States', fontsize=20, fontweight='bold', pad=15)
axes[0].legend(fontsize=14, frameon=True, shadow=True)
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(labelsize=14)

# Impact of jump_penalty (log scale)
for model in grid_df['model_name'].unique():
    model_data = grid_df[grid_df['model_name'] == model]
    # Bin jump penalties for clarity
    model_data['penalty_bin'] = pd.cut(model_data['jump_penalty'], bins=5)
    penalty_stats = model_data.groupby('penalty_bin')['balanced_accuracy'].agg(['mean', 'std'])
    
    x_vals = range(len(penalty_stats))
    axes[1].errorbar(x_vals, penalty_stats['mean'], 
                     yerr=penalty_stats['std'], 
                     marker='s', markersize=10, linewidth=3,
                     capsize=6, capthick=2, label=model)

axes[1].set_xlabel('Jump Penalty (λ) - Binned', fontsize=18, fontweight='bold')
axes[1].set_ylabel('Balanced Accuracy', fontsize=18, fontweight='bold')
axes[1].set_title('Impact of Jump Penalty', fontsize=20, fontweight='bold', pad=15)
axes[1].legend(fontsize=14, frameon=True, shadow=True)
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(labelsize=14)
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(['Very Low', 'Low', 'Med', 'High', 'Very High'], 
                        fontsize=13, rotation=0)

# Impact of max_feats
for model in grid_df['model_name'].unique():
    model_data = grid_df[grid_df['model_name'] == model]
    # Bin max_feats for clarity
    model_data['feats_bin'] = pd.cut(model_data['max_feats'], bins=5)
    feats_stats = model_data.groupby('feats_bin')['balanced_accuracy'].agg(['mean', 'std'])
    
    x_vals = range(len(feats_stats))
    axes[2].errorbar(x_vals, feats_stats['mean'], 
                     yerr=feats_stats['std'], 
                     marker='^', markersize=10, linewidth=3,
                     capsize=6, capthick=2, label=model)

axes[2].set_xlabel('Max Features (κ²) - Binned', fontsize=18, fontweight='bold')
axes[2].set_ylabel('Balanced Accuracy', fontsize=18, fontweight='bold')
axes[2].set_title('Impact of Feature Budget', fontsize=20, fontweight='bold', pad=15)
axes[2].legend(fontsize=14, frameon=True, shadow=True)
axes[2].grid(True, alpha=0.3)
axes[2].tick_params(labelsize=14)
axes[2].set_xticks(range(5))
axes[2].set_xticklabels(['Very Low', 'Low', 'Med', 'High', 'Very High'], 
                        fontsize=13, rotation=0)

plt.tight_layout()
plt.savefig(output_dir / "05_hyperparameter_impact.jpg", 
            dpi=300, bbox_inches='tight', format='jpg')
plt.close()
print(f"   ✓ Saved: {output_dir / '05_hyperparameter_impact.jpg'}")

##############################################################################
# Figure 6: Summary Statistics Table (as image)
##############################################################################
print("\n6. Creating Summary Statistics Table...")

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

# Compute summary statistics
summary_stats = []
for model in results_df['model_name'].unique():
    model_data = results_df[results_df['model_name'] == model]
    summary_stats.append({
        'Model': model,
        'BAC (mean)': f"{model_data['balanced_accuracy'].mean():.4f}",
        'BAC (std)': f"{model_data['balanced_accuracy'].std():.4f}",
        'F1 (mean)': f"{model_data['feature_f1'].mean():.4f}",
        'F1 (std)': f"{model_data['feature_f1'].std():.4f}",
        'Precision': f"{model_data['feature_precision'].mean():.4f}",
        'Recall': f"{model_data['feature_recall'].mean():.4f}",
        'Time (s)': f"{model_data['computation_time'].mean():.2f}",
        'N': len(model_data)
    })

summary_df = pd.DataFrame(summary_stats)

# Create table
table = ax.table(cellText=summary_df.values,
                colLabels=summary_df.columns,
                cellLoc='center',
                loc='center',
                bbox=[0, 0, 1, 1])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(16)
table.scale(1, 3)

# Header row styling
for i in range(len(summary_df.columns)):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white', fontsize=18)
    cell.set_height(0.12)

# Data row styling
for i in range(1, len(summary_df) + 1):
    for j in range(len(summary_df.columns)):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#E7E6E6')
        else:
            cell.set_facecolor('#F2F2F2')
        cell.set_text_props(fontsize=16)
        cell.set_height(0.1)

ax.set_title('Summary Statistics: Model Performance Comparison', 
             fontsize=24, fontweight='bold', pad=40)

plt.tight_layout()
plt.savefig(output_dir / "06_summary_table.jpg", 
            dpi=300, bbox_inches='tight', format='jpg')
plt.close()
print(f"   ✓ Saved: {output_dir / '06_summary_table.jpg'}")

##############################################################################
# Figure 7: Boxplot Comparison
##############################################################################
print("\n7. Creating Boxplot Comparison...")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Boxplot for Balanced Accuracy
bp1 = axes[0].boxplot([results_df[results_df['model_name'] == m]['balanced_accuracy'].values 
                        for m in results_df['model_name'].unique()],
                       labels=results_df['model_name'].unique(),
                       patch_artist=True,
                       widths=0.6)

# Color the boxes
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_linewidth(2)

# Style whiskers, caps, medians
for element in ['whiskers', 'caps', 'medians']:
    for item in bp1[element]:
        item.set_linewidth(2)
        if element == 'medians':
            item.set_color('red')
            item.set_linewidth(3)

axes[0].set_ylabel('Balanced Accuracy', fontsize=20, fontweight='bold')
axes[0].set_title('Distribution of Balanced Accuracy', 
                  fontsize=22, fontweight='bold', pad=20)
axes[0].tick_params(labelsize=16)
axes[0].grid(True, alpha=0.3, axis='y')

# Boxplot for Feature F1
bp2 = axes[1].boxplot([results_df[results_df['model_name'] == m]['feature_f1'].values 
                        for m in results_df['model_name'].unique()],
                       labels=results_df['model_name'].unique(),
                       patch_artist=True,
                       widths=0.6)

# Color the boxes
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_linewidth(2)

# Style whiskers, caps, medians
for element in ['whiskers', 'caps', 'medians']:
    for item in bp2[element]:
        item.set_linewidth(2)
        if element == 'medians':
            item.set_color('red')
            item.set_linewidth(3)

axes[1].set_ylabel('Feature Selection F1', fontsize=20, fontweight='bold')
axes[1].set_title('Distribution of Feature Selection F1', 
                  fontsize=22, fontweight='bold', pad=20)
axes[1].tick_params(labelsize=16)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / "07_boxplot_comparison.jpg", 
            dpi=300, bbox_inches='tight', format='jpg')
plt.close()
print(f"   ✓ Saved: {output_dir / '07_boxplot_comparison.jpg'}")

##############################################################################
# Print Summary
##############################################################################
print("\n" + "="*80)
print("Visualization Complete!")
print("="*80)
print(f"\nAll figures saved to: {output_dir}/")
print("\nGenerated figures:")
print("  1. 01_model_comparison_bac.jpg - Bar chart of balanced accuracy")
print("  2. 02_model_comparison_f1.jpg - Bar chart of feature selection F1")
print("  3. 03_performance_by_delta.jpg - Performance vs. separation level")
print("  4. 04_computation_time.jpg - Computational efficiency")
print("  5. 05_hyperparameter_impact.jpg - Hyperparameter sensitivity")
print("  6. 06_summary_table.jpg - Summary statistics table")
print("  7. 07_boxplot_comparison.jpg - Distribution comparison")
print("\nAll figures are:")
print("  - High resolution (300 DPI)")
print("  - JPEG format for report inclusion")
print("  - Large, readable text (16-22pt)")
print("  - Publication quality")
