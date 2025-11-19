# Sparse Jump Model - Divergence Support Update

## Summary

The `SparseJumpModel` class has been updated to support different divergence measures beyond the traditional squared Euclidean distance. This allows the model to handle different data types more appropriately (e.g., count data with Poisson KL divergence).

## Changes Made

### 1. New Parameter: `distribution`

Added a `distribution` parameter to `SparseJumpModel.__init__()`:

```python
SparseJumpModel(
    n_components=2,
    max_feats=100.,
    jump_penalty=0.,
    distribution="Gaussian",  # NEW: "Gaussian" or "Poisson"
    cont=False,
    ...
)
```

**Options:**
- `"Gaussian"` (default): Uses squared Euclidean distance (L2)
- `"Poisson"`: Uses Poisson KL divergence for count data

### 2. Updated `compute_BCSS()` Function

The function now computes different measures based on the distribution:

#### Gaussian (Original Behavior)
- Computes traditional Between Cluster Sum of Squares (BCSS)
- Formula: `BCSS = Σ_k N_k * (μ_k - μ_global)²`

#### Poisson (New)
- Computes between-state variation based on KL divergence
- Formula from your algorithm:
  ```
  a_j = Σ_t [μ̄_j - y_{t,j}(1 + log(μ̄_j/y_{t,j}))] 
      - Σ_t [μ_{s_t,j} - y_{t,j}(1 + log(μ_{s_t,j}/y_{t,j}))]
  ```

### 3. Updated Algorithm Flow

The sparse jump model now follows the updated algorithm you provided:

**For Gaussian distribution:**
- Uses weighted squared Euclidean distance
- Computes centroids via weighted mean
- Updates weights based on BCSS

**For Poisson distribution:**
- Uses weighted Poisson KL divergence
- Computes centroids via mean of assigned samples
- Updates weights based on between-state KL variation

## Usage Examples

### Example 1: Gaussian (Traditional)

```python
from jumpmodels import SparseJumpModel

# For continuous/Gaussian data
model_gaussian = SparseJumpModel(
    n_components=3,
    max_feats=25,
    jump_penalty=50.0,
    distribution="Gaussian",  # Default
    verbose=1
)

model_gaussian.fit(X_continuous_data)
labels = model_gaussian.predict(X_test)
```

### Example 2: Poisson (Count Data)

```python
from jumpmodels import SparseJumpModel

# For count data (e.g., transaction counts, event frequencies)
model_poisson = SparseJumpModel(
    n_components=3,
    max_feats=25,
    jump_penalty=50.0,
    distribution="Poisson",  # NEW
    verbose=1
)

model_poisson.fit(X_count_data)
labels = model_poisson.predict(X_test)
```

## Mathematical Details

### Poisson KL Divergence

The Poisson KL divergence between observation `y` and rate `μ` is:

```
D_KL(y || μ) = μ - y * log(μ) + log(y!) 
             ≈ μ - y * (1 + log(μ/y))  [ignoring constant log(y!)]
```

### Between-State Variation (Poisson)

For feature `j`, the between-state variation measures how much the cluster assignments reduce the KL divergence from the global mean:

```
a_j = Total_divergence_j - Within_cluster_divergence_j
```

Where:
- `Total_divergence_j`: Sum of KL divergences from global mean `μ̄_j`
- `Within_cluster_divergence_j`: Sum of KL divergences from assigned cluster means `μ_{s_t,j}`

Features with higher `a_j` values are more important for distinguishing between states.

## Implementation Notes

1. **Numerical Stability**: Added safeguards against `log(0)` by clamping values to minimum of `1e-10`

2. **Hard vs Soft Assignment**: The Poisson BCSS computation uses hard assignment (`argmax`) for computational efficiency, consistent with the K-means step in the original algorithm

3. **Backward Compatibility**: The default `distribution="Gaussian"` ensures existing code continues to work unchanged

4. **Distribution Propagation**: The distribution parameter is automatically passed to the underlying `JumpModel` instance

## Future Extensions

To add more divergence measures (e.g., Negative Binomial):

1. Update `do_E_step()` in `jump.py` to handle the new distribution
2. Add the corresponding between-state variation computation in `compute_BCSS()`
3. Add the distribution name to the docstrings

Example structure:
```python
elif distribution == "NegativeBinomial":
    # Compute NB KL divergence-based variation
    # ...
```

## Testing Recommendations

1. **Synthetic Count Data**: Test with simulated Poisson data with known cluster structure
2. **Real Count Data**: Apply to transaction data, event counts, etc.
3. **Comparison**: Compare Gaussian vs Poisson on same dataset to see which performs better
4. **Edge Cases**: Test with data containing zeros, very small/large counts

## References

- Original Sparse Jump Model: Nystrup et al. (2021)
- Poisson KL Divergence: Your provided algorithm
- Soft Thresholding: Witten et al. (2010)
