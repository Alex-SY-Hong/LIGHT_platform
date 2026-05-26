# LLM Consensus Analysis Modules

This document describes how to use the modularized analysis interfaces for LLM Consensus.

## Overview

The project has been modularized to provide clean, reusable interfaces for:

1. **Popularity Bias Analysis** (`popularity_bias/`) - Robust regression-based analysis
2. **Anti-Formula Analysis** (`anti_analysis/`) - Extract arguments against specific formulas
3. **Unified API** (`analyze.py`) - Single entry point for all analyses

---

## Module 1: Popularity Bias Analysis

### What it does

Detects and corrects popularity bias in LLM scoring using:
- **Partial Correlation** - Controls for confounding variables
- **Robust Regression** - Huber + RANSAC ensemble for outlier resistance

### Method A: Use the convenience function

```python
from popularity_bias import analyze_popularity_bias

# Run analysis with auto-detected paths
results = analyze_popularity_bias()

# Run with custom paths
results = analyze_popularity_bias(
    data_dir="./my_custom_data",
    results_dir="./my_custom_results",
    configure_logging=False  # Disable logging
)

# Access results
for model_name, model_results in results.items():
    optimal_formula = model_results['optimal_formula']
    print(f"{model_name}: Formula {optimal_formula['formula_id']} ({optimal_formula['formula_name']})")
    print(f"  Total Score: {optimal_formula['total_score']:.2f}")
```

### Method B: Use the analyzer class directly

```python
from popularity_bias import RigorousBiasAnalyzer

# Create analyzer
analyzer = RigorousBiasAnalyzer(
    project_root="/path/to/project",  # Optional, auto-detected by default
    data_dir="./popularity_bias/data",  # Optional, uses default
    results_dir="./popularity_bias/results"  # Optional, uses default
)

# Run analysis
results = analyzer.run_rigorous_analysis()

# Access specific methods
rho, p_val, needs_debias = analyzer.calculate_partial_correlation(
    model="gpt-5",
    dimension="Mechanical_Safety",
    relative_frequencies=analyzer.relative_frequencies
)
```

### Command Line Interface

```bash
# Run popularity bias analysis
python analyze.py popularity

# With custom directories
python analyze.py popularity --data-dir ./data --results-dir ./results

# Without logging
python analyze.py popularity --no-logging
```

---

## Module 2: Anti-Formula Analysis

### What it does

Extracts arguments against specific formulas from LLM output files.

### Method A: Use the convenience function

```python
from anti_analysis import extract_anti_arguments

# Extract arguments against Formula 4 and 5 (default)
arguments = extract_anti_arguments()

# Extract for specific formulas only
arguments = extract_anti_arguments(
    formulas=[4]  # Only Formula 4
)

# With custom model files
model_files = {
    "gpt-5": "gpt-5.md",
    "grok-4": "grok-4.md"
}
arguments = extract_anti_arguments(model_files=model_files)

# With custom output directory
arguments = extract_anti_arguments(output_dir="./custom_output")

# Access results
for formula_num, args in arguments.items():
    print(f"Formula {formula_num}: {len(args)} arguments found")
    for arg in args[:3]:  # Print first 3 arguments
        print(f"  - {arg}")
```

### Method B: Use the extractor class directly

```python
from anti_analysis import AntiArgumentExtractor

# Create extractor
extractor = AntiArgumentExtractor(
    model_files={  # Optional, uses defaults
        "gpt-5": "gpt-5.md",
        "grok-4": "grok-4.md"
    },
    project_root="/path/to/project",  # Optional, auto-detected
    output_dir="./anti_analysis/results"  # Optional, uses default
)

# Extract arguments
arguments = extractor.extract(formulas=[4, 5])

# Or use individual methods
runs = extractor.extract_runs_from_file("gpt-5.md")
args_4 = extractor.extract_formula_arguments(runs[0][1], 4)
args_5 = extractor.extract_from_rejected_section(runs[0][1], 5)
```

### Command Line Interface

```bash
# Run anti-formula analysis (default: Formula 4 and 5)
python analyze.py anti

# Extract for specific formulas only
python analyze.py anti --formulas 4

# With custom output directory
python analyze.py anti --output-dir ./custom_output
```

---

## Unified API: analyze.py

### What it does

Provides a single entry point for running all available analyses.

### Command Line Interface

```bash
# Show help
python analyze.py --help

# Run popularity bias analysis only
python analyze.py popularity

# Run anti-formula analysis only
python analyze.py anti

# Run all analyses
python analyze.py all

# Run with custom directories
python analyze.py popularity --data-dir ./data --results-dir ./results

# Run anti-formula with specific formulas
python analyze.py anti --formulas 4 5

# Run all with custom settings
python analyze.py all \
    --popularity-data-dir ./data \
    --popularity-results-dir ./results \
    --anti-results-dir ./anti_output \
    --formulas 4 5
```

### Python API

```python
from analyze import AnalysisPipeline

# Create pipeline
pipeline = AnalysisPipeline(project_root="/path/to/project")

# Run popularity bias analysis
popularity_results = pipeline.run_popularity_bias_analysis()

# Run anti-formula analysis
anti_results = pipeline.run_anti_formula_analysis(
    formulas=[4, 5],
    output_dir="./custom_output"
)

# Run all analyses
all_results = pipeline.run_all(
    popularity_data_dir="./data",
    popularity_results_dir="./results",
    anti_results_dir="./anti_output",
    formulas=[4, 5],
    configure_logging=True
)

# Access all results
print(all_results['popularity_bias'])
print(all_results['anti_formula'])
```

---

## Directory Structure

```
LLM_consensus/
├── analyze.py                      # Unified API entry point
├── popularity_bias/                # Popularity bias module
│   ├── __init__.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── robust_regression.py    # Main analysis logic
│   ├── data/                       # Input data
│   │   ├── material_frequencies.json
│   │   ├── relative_frequencies.json
│   │   ├── formula_materials.json
│   │   └── extracted_data.json
│   ├── results/                    # Output results
│   │   ├── *_debiased_rigorous_v2.json
│   │   └── rigorous_analysis_v2_summary.json
│   └── scripts/                    # Legacy scripts (optional)
├── anti_analysis/                  # Anti-formula module
│   ├── __init__.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── extract_arguments.py     # Main extraction logic
│   ├── results/                    # Output results
│   │   ├── anti-Formula 4.md
│   │   └── anti-Formula 5.md
│   └── scripts/                    # Legacy scripts (optional)
└── [other project files...]
```

---

## Example Workflows

### Example 1: Quick Analysis

```bash
# Run popularity bias analysis
python analyze.py popularity
```

### Example 2: Complete Analysis Pipeline

```bash
# Run all analyses
python analyze.py all
```

### Example 3: Custom Analysis in Python

```python
from analyze import AnalysisPipeline

pipeline = AnalysisPipeline()

# Run popularity bias analysis with custom settings
popularity_results = pipeline.run_popularity_bias_analysis(
    data_dir="./custom_data",
    results_dir="./custom_results",
    configure_logging=False
)

# Extract anti-formula arguments for specific formulas
anti_results = pipeline.run_anti_formula_analysis(
    formulas=[4],  # Only Formula 4
    output_dir="./custom_output"
)
```

### Example 4: Using individual modules

```python
# Popularity bias analysis
from popularity_bias import RigorousBiasAnalyzer

analyzer = RigorousBiasAnalyzer()
results = analyzer.run_rigorous_analysis()

# Anti-formula analysis
from anti_analysis import AntiArgumentExtractor

extractor = AntiArgumentExtractor()
arguments = extractor.extract(formulas=[5])  # Only Formula 5
```

---

## Migration from Old Scripts

### Old way (direct script execution):

```bash
cd popularity_bias/scripts
python analyze_rigorous_v2.py
cd ../../anti_analysis/scripts
python extract_anti_arguments.py
```

### New way (unified interface):

```bash
# Popularity bias analysis
python analyze.py popularity

# Anti-formula analysis
python analyze.py anti

# Or both at once
python analyze.py all
```

---

## Error Handling

### Common Issues

1. **Module import error**
   ```python
   # Make sure you're in the project root
   import os
   os.chdir('/path/to/LLM_consensus')
   from popularity_bias import analyze_popularity_bias
   ```

2. **Data not found**
   ```python
   # Specify custom data directory
   results = analyze_popularity_bias(data_dir="./path/to/data")
   ```

3. **Output directory permissions**
   ```python
   # Specify writable output directory
   results = analyze_popularity_bias(results_dir="./writable_output")
   ```

---

## API Reference

### popularity_bias.analyze_popularity_bias()

```python
def analyze_popularity_bias(
    project_root: str = None,
    data_dir: str = None,
    results_dir: str = None,
    configure_logging: bool = True
) -> Dict:
    """
    Convenience function to run popularity bias analysis.

    Args:
        project_root: Root directory of the project (default: auto-detect)
        data_dir: Directory containing input data
        results_dir: Directory for output results
        configure_logging: Whether to configure logging

    Returns:
        Dictionary containing full analysis results
    """
```

### anti_analysis.extract_anti_arguments()

```python
def extract_anti_arguments(
    model_files: Dict[str, str] = None,
    project_root: str = None,
    output_dir: str = None,
    formulas: List[int] = None
) -> Dict[int, List[str]]:
    """
    Convenience function to extract arguments against specified formulas.

    Args:
        model_files: Dictionary mapping model names to their output files
        project_root: Root directory of the project
        output_dir: Directory for output results
        formulas: List of formula numbers to extract arguments for

    Returns:
        Dictionary mapping formula numbers to lists of unique arguments
    """
```

### AnalysisPipeline

```python
class AnalysisPipeline:
    """Unified pipeline for running multiple analyses."""

    def __init__(self, project_root: str = None):
        """Initialize the analysis pipeline."""

    def run_popularity_bias_analysis(...) -> Dict:
        """Run popularity bias analysis using robust regression."""

    def run_anti_formula_analysis(...) -> Dict[int, List[str]]:
        """Run anti-formula analysis to extract arguments."""

    def run_all(**kwargs) -> Dict:
        """Run all available analyses."""
```

---

## Support

For issues or questions, please refer to:
- `DIRECTORY_STRUCTURE.md` - Project directory structure
- Original script files in `scripts/` directories for reference