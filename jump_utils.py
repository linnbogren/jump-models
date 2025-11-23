import numpy as np
import pandas as pd
from jumpmodels.jump import JumpModel
from jumpmodels.sparse_jump import SparseJumpModel 
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_regime_changes(X, model, actual_breakpoints=None, title='Model Results'):
    """
    Plots the results of a fitted JumpModel, showing both predicted
    and actual regime change points.

    Parameters:
    - X (pd.DataFrame): The input time series data.
    - model (JumpModel): The fitted jump model object.
    - actual_breakpoints (int, list, or np.ndarray, optional): A single value or a
      list of the true breakpoint indices to plot for comparison.
    - title (str): The title for the plot.
    """
    # --- ADDED LOGIC ---
    # Normalize input to always be a list if it's a single number
    if actual_breakpoints is not None and isinstance(actual_breakpoints, (int, float)):
        actual_breakpoints = [actual_breakpoints]
    # --------------------

    labels = model.labels_.to_numpy()
    
    # Find the integer positions of the PREDICTED change points
    change_point_indices = np.where(labels[:-1] != labels[1:])[0] + 1

    # Convert integer indices to the actual index values (e.g., dates)
    if isinstance(X.index, pd.DatetimeIndex):
        change_point_values = X.index[change_point_indices]
        # Handle conversion for a list of actual breakpoints
        if actual_breakpoints is not None:
            valid_indices = [bp for bp in actual_breakpoints if bp < len(X.index)]
            breakpoint_vals = X.index[valid_indices]
        else:
            breakpoint_vals = []
    else:
        change_point_values = change_point_indices
        breakpoint_vals = actual_breakpoints if actual_breakpoints is not None else []
            
    features = X.columns
    n_features = len(features)

    # Use a single axis for a single feature plot
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 4 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes] # Make it iterable
    fig.suptitle(title, fontsize=18)

    base_colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum', 'khaki']
    
    for i, feature in enumerate(features):
        ax = axes[i]
        colors = [base_colors[i % len(base_colors)]]

        ax.plot(X.index, X[feature], label=f'{feature} Data', color=colors[0], zorder=2)
        
        # Loop to plot multiple ACTUAL breakpoints
        for j, bp in enumerate(breakpoint_vals):
            # Only add a label for the very first line to avoid legend clutter
            label_actual = 'Actual Breakpoint' if i == 0 and j == 0 else ""
            ax.axvline(x=bp, color='green', linestyle='-', linewidth=2.5, label=label_actual, zorder=3)
        
        # Loop to plot multiple PREDICTED change points
        for j, cp in enumerate(change_point_values):
            # Only add a label for the very first line
            label_pred = 'Predicted Change Point' if i == 0 and j == 0 else ""
            ax.axvline(x=cp, color='crimson', linestyle='--', linewidth=2, label=label_pred, zorder=3)

        ax.set_ylabel('Value', fontsize=12)
        title_text = f'Analysis of {feature.capitalize()}'
        
        # Check for and display feature weights if they exist
        if hasattr(model, 'feat_weights') and model.feat_weights is not None:
            weight = model.feat_weights.iloc[i] if isinstance(model.feat_weights, pd.Series) else model.feat_weights[i]
            title_text += f" (Feature Weight: {weight:.4f})"
        
        ax.set_title(title_text, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    axes[-1].set_xlabel('Time Step', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

def plot_simulated_from_regimes(X, model, title='Simulated Data from Predicted Regimes'):
    """
    Generates and plots a new time series by sampling from the distributions
    of the predicted states from a fitted JumpModel.

    Parameters:
    - X (pd.DataFrame): The original data, used for index, columns, and shape.
    - model (JumpModel): The fitted jump model object.
    - title (str): The title for the plot.
    """
    # --- 1. Extract model parameters ---
    labels = model.labels_.to_numpy()
    centers = model.centers_  # These are the means (lambda for Poisson, mu for Gaussian)
    distribution = model.distribution
    n_samples, n_features = X.shape
    
    # For Gaussian, we need to estimate the standard deviation for each state
    stds = None
    if distribution == "Gaussian":
        stds = np.array([X[labels == i].std(axis=0).fillna(1.0) for i in range(model.n_components)])

    # --- 2. Simulate data from the predicted regimes ---
    simulated_data = np.zeros_like(X)
    for t in range(n_samples):
        state = labels[t]
        params = centers[state] # Mean parameters for the current state

        if distribution == "Poisson":
            # Sample from Poisson distribution for each feature
            simulated_data[t, :] = np.random.poisson(lam=params)
        elif distribution == "Gaussian":
            # Sample from Gaussian distribution for each feature
            state_stds = stds[state]
            simulated_data[t, :] = np.random.normal(loc=params, scale=state_stds)
        else:
            raise NotImplementedError(f"Simulation not implemented for '{distribution}' distribution.")

    simulated_df = pd.DataFrame(simulated_data, index=X.index, columns=X.columns)

    # --- 3. Prepare for Plotting ---
    change_point_indices = np.where(labels[:-1] != labels[1:])[0] + 1
    if isinstance(X.index, pd.DatetimeIndex):
        change_point_values = X.index[change_point_indices]
    else:
        change_point_values = change_point_indices

    # --- 4. Create the Plot ---
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 4 * n_features), sharex=True)
    if n_features == 1:
        axes = [axes] # Make it iterable
    fig.suptitle(title, fontsize=18)

    base_colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum', 'khaki']

    for i, feature in enumerate(simulated_df.columns):
        ax = axes[i]
        color = base_colors[i % len(base_colors)]

        # Plot the NEWLY SIMULATED data
        ax.plot(simulated_df.index, simulated_df[feature], label=f'Simulated {feature}', color=color, zorder=2)
        
        # Loop to plot the predicted change points
        for j, cp in enumerate(change_point_values):
            label_pred = 'Predicted Change Point' if i == 0 and j == 0 else ""
            ax.axvline(x=cp, color='crimson', linestyle='--', linewidth=2, label=label_pred, zorder=3)

        ax.set_ylabel('Simulated Value', fontsize=12)
        ax.set_title(f'Simulated Data for {feature.capitalize()} based on Predicted Regimes', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    axes[-1].set_xlabel('Time Step', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


def plot_stacked_states(X, models_dict, true_states=None, feature_to_plot=None, figsize=(12, 10)):
    """
    Creates a stacked plot showing time series data and state assignments from multiple models.
    
    Similar to the visualization in academic papers, this shows:
    - Top panel: The time series data (one or more features)
    - Subsequent panels: State assignments from different models as horizontal bars
    
    Parameters:
    -----------
    X : pd.DataFrame
        The input data.
    models_dict : dict
        Dictionary of {model_name: model} where each model has a labels_ attribute.
        Example: {'Jump Model': jm, 'Sparse Jump': sjm, 'Poisson Jump': pjm}
    true_states : np.ndarray or pd.Series, optional
        The true state sequence if available.
    feature_to_plot : str or list, optional
        Name(s) of feature(s) to plot. If None, plots all features.
    figsize : tuple
        Figure size (width, height).
    """
    # Determine which features to plot
    if feature_to_plot is None:
        features = X.columns.tolist()
    elif isinstance(feature_to_plot, str):
        features = [feature_to_plot]
    else:
        features = feature_to_plot
    
    # Calculate number of panels needed
    n_data_panels = len(features)
    n_model_panels = len(models_dict)
    n_panels = n_data_panels + n_model_panels
    if true_states is not None:
        n_panels += 1
    
    # Create the figure with subplots
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]
    
    fig.suptitle('Time Series and State Assignments', fontsize=16, y=0.995)
    
    # Color palette for data
    data_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    panel_idx = 0
    
    # Plot the time series data
    for i, feature in enumerate(features):
        ax = axes[panel_idx]
        ax.plot(X.index, X[feature], color=data_colors[i % len(data_colors)], 
                linewidth=1.5, alpha=0.8)
        ax.set_ylabel(feature, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='y', labelsize=9)
        panel_idx += 1
    
    # Function to plot state bars
    def plot_state_bars(ax, labels, label_text, color_map=None):
        """Helper function to plot horizontal state bars."""
        labels_array = labels.to_numpy() if hasattr(labels, 'to_numpy') else np.array(labels)
        n_states = len(np.unique(labels_array))
        
        # Default color map if none provided
        if color_map is None:
            # Use a colormap (updated for matplotlib 3.7+)
            cmap = plt.colormaps.get_cmap('Set2')
            color_map = {i: cmap(i / max(n_states - 1, 1)) for i in range(n_states)}
        
        # Plot horizontal bars for each state
        current_state = labels_array[0]
        start_idx = 0
        
        for t in range(1, len(labels_array) + 1):
            # Check if we've reached the end or if state changed
            if t == len(labels_array) or labels_array[t] != current_state:
                # Plot bar for the previous segment
                # Use state value as y-position (1, 2, 3, etc.)
                y_pos = current_state + 1  # +1 to make it 1-indexed for display
                ax.barh(y=y_pos, width=t - start_idx, left=start_idx, 
                       height=0.8, color=color_map[current_state], 
                       edgecolor='white', linewidth=0.5, align='center')
                
                if t < len(labels_array):
                    current_state = labels_array[t]
                    start_idx = t
        
        ax.set_ylabel(label_text, fontsize=10, rotation=0, ha='right', va='center')
        ax.set_ylim(0.5, n_states + 0.5)
        ax.set_yticks(range(1, n_states + 1))
        ax.set_yticklabels([f'State {i}' for i in range(1, n_states + 1)], fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.tick_params(left=True, labelsize=8)
    
    # Plot true states if provided
    if true_states is not None:
        ax = axes[panel_idx]
        plot_state_bars(ax, true_states, 'True\nStates')
        panel_idx += 1
    
    # Plot model predictions
    for model_name, model in models_dict.items():
        ax = axes[panel_idx]
        plot_state_bars(ax, model.labels_, model_name)
        panel_idx += 1
    
    # Set x-axis label on the bottom panel
    axes[-1].set_xlabel('Time Step', fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.15)
    plt.show()
