# Refactoring Guidelines for Popularity Bias Analysis Pipeline

## Context & Objective
The current codebase utilizes an extreme Object-Oriented "God Class" (`PopularityBiasAnalyzer`) that handles API requests, file I/O, data transformation, and statistical calculations all at once. Furthermore, there are critical statistical flaws in how the variance and bias are handled. 

Your objective is to refactor this script into a **Hybrid Functional-OOP architecture** and implement the **statistically rigorous math logic** detailed below.

---

## 1. Architectural Paradigm Shift (FP + OOP)
Do not use a single God Class. Deconstruct the existing pipeline into three distinct components:

1. **`ExposureAPIClient` (OOP for State & Side-Effects):**
   - Create a dedicated class that ONLY handles `aiohttp` sessions, API rate limiting (`asyncio.Semaphore`), retries, and caching.
   - It should have methods like `fetch_datamuse`, `fetch_arxiv`, `fetch_pubchem_synonyms`.
2. **`StatisticalEngine` (Pure Functions for Math & Flow):**
   - Extract mathematical operations into pure functions (no `self` dependencies).
   - Data in, Data out. These functions must be deterministic and side-effect free.
3. **`PipelineOrchestrator` (The Main Script):**
   - A function that instantiates the API client, calls the APIs, passes data to the pure functions, and handles file saving (JSON/CSV outputs).

---

## 2. CRITICAL Statistical Logic Constraints
When rewriting the mathematical functions, you MUST adhere to the following statistical rules. **Failure to do so invalidates the scientific integrity of the research.**

*   **Rule A (Denoise before Correlation):** Never calculate Spearman correlation on raw multi-run data (which contains aleatoric noise). You MUST first aggregate the data using `groupby('Formula').mean()` to get the true score signal before correlating with frequencies.
*   **Rule B (Dual-Criterion Threshold):** Do not use a hardcoded `abs(rho) > 0.5`. You MUST use: `abs(rho) > 0.3 AND p_value < 0.10`.
*   **Rule C (Isolate Total_Score):** `Total_Score` must NOT be directly regressed or debiased. Only debias the 6 specific evaluation dimensions. The final adjusted `Total_Score` must be dynamically recalculated by summing the debiased sub-dimensions.
*   **Rule D (Exception Handling):** Do not use naked `except:`. Explicitly catch `ValueError` and `Exception` inside the SciPy correlation block to prevent swallowing system exits while ensuring pipeline robustness.

---

## 3. Code Implementation for Core Statistical Functions
Use the following pure functions to replace the existing mathematical methods. Ensure you pass the `dimensions` list (excluding `Total_Score`) into them.

### Function 1: Spearman Correlation
```python
import pandas as pd
import numpy as np
import logging
from scipy.stats import spearmanr
from typing import Dict, List

logger = logging.getLogger(__name__)

def calculate_spearman_correlations(
    scores_df: pd.DataFrame, 
    relative_frequencies: Dict[int, float],
    dimensions: List[str]
) -> Dict:
    """
    Pure function to calculate correlations.
    Rule applied: Denoise first (Mean Aggregation), then Correlate.
    """
    correlation_results = {}
    
    # Step 1: Mean Aggregation (Denoising Aleatoric Variance)
    mean_scores_df = scores_df.groupby('Formula')[dimensions].mean().reset_index()

    for dimension in dimensions:
        mean_scores = mean_scores_df[dimension].values
        freqs = np.array([relative_frequencies[f] for f in mean_scores_df['Formula']])

        try:
            rho, p_value = spearmanr(mean_scores, freqs)
            # Handle cases where variance is 0 (constant scores return NaN)
            if np.isnan(rho) or np.isnan(p_value):
                rho, p_value = 0.0, 1.0
                
        except ValueError as e:
            logger.warning(f"  [Warning] Spearmanr calculation failed for {dimension} (ValueError): {e}. Defaulting to 0.0")
            rho, p_value = 0.0, 1.0
        except Exception as e:
            logger.error(f"  [Error] Unexpected failure in Spearmanr for {dimension}: {e}. Defaulting to 0.0")
            rho, p_value = 0.0, 1.0

        # Step 3: Dual-Criterion Thresholding
        # Moderate effect size (> 0.3) AND statistical significance (p < 0.10)
        needs_debiasing = bool(abs(rho) > 0.3 and p_value < 0.10)

        correlation_results[dimension] = {
            "rho": float(rho),
            "p_value": float(p_value),
            "needs_debiasing": needs_debiasing
        }

    return correlation_results
```

### Function 2: Mean-Anchoring Debiasing
```python
from sklearn.linear_model import LinearRegression

def apply_mean_anchoring_debias(
    scores_df: pd.DataFrame, 
    relative_frequencies: Dict[int, float], 
    correlation_results: Dict,
    dimensions: List[str]
) -> pd.DataFrame:
    """
    Pure function to debias scores using Mean-Anchoring shift based on denoised data.
    """
    # Step 1: Denoise first. Apply debiasing to the aggregated MEAN scores.
    df_mean = scores_df.groupby('Formula')[dimensions].mean().reset_index()
    df_mean['relative_freq'] = df_mean['Formula'].map(relative_frequencies)
    
    debiased_df = pd.DataFrame({'Formula': df_mean['Formula']})

    for dimension in dimensions:
        if correlation_results[dimension]['needs_debiasing']:
            X = df_mean['relative_freq'].values.reshape(-1, 1)
            y = df_mean[dimension].values

            # Linear regression to get predicted bias
            model_lr = LinearRegression()
            model_lr.fit(X, y)
            y_pred = model_lr.predict(X)

            # Extract Residuals
            residuals = y - y_pred

            # Calculate global mean of raw LLM scores
            mean_y = np.mean(y)

            # Mean-anchoring shift
            debiased = residuals + mean_y

            # Clip to 0-10 range
            debiased_df[f"{dimension}_debiased"] = np.clip(debiased, 0, 10)
        else:
            # Keep raw mean score if no systemic bias detected
            debiased_df[f"{dimension}_debiased"] = df_mean[dimension].values

    # Step 3: Recalculate TRUE Adjusted Total Score dynamically
    debiased_columns = [f"{dim}_debiased" for dim in dimensions]
    debiased_df['Total_Score_debiased'] = debiased_df[debiased_columns].sum(axis=1)

    return debiased_df
```

## Action Items for Claude Code:
1. Parse the existing `PopularityBiasAnalyzer` file.
2. Extract the API logic into `ExposureAPIClient` class.
3. Replace the math logic with the pure functions provided above.
4. Wire them together in a clean `main()` orchestrator script that iterates through the independent LLM models.
5. Retain all existing JSON/logging I/O functionalities but manage them at the orchestrator level.
