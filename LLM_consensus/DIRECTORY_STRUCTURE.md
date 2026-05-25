# Directory Structure

This document describes the reorganized directory structure for the LLM Consensus - Popularity Bias Analysis project.

## Root Directory

The root directory contains the main project files:

- **analyze.py** - Unified API interface for all analyses (NEW)
- **llm_concensus.py** - Main script for running LLM consensus analysis
- **extract_data.py** - Data extraction script (generates `extracted_data.json`)
- **analyze_llm_reliability.py** - Reliability analysis script
- **generate_tex_report.py** / **generate_tex_report_en.py** - LaTeX report generation scripts
- **example_visualization.py** - Example visualization script
- **run_pipeline.py** - Pipeline execution script
- **test_optimal_formula.py** - Test script for optimal formula selection
- **fix_quotes.py** - Utility script for fixing quotes

### LLM Output Files

- `gpt-5.md` - GPT-5 output (11 runs)
- `grok-4.md` - Grok-4 output (11 runs)
- `claude-opus-4-5-20251101.md` - Claude Opus 4.5 output (11 runs)
- `gemini-3-pro-preview.md` - Gemini 3 Pro output (11 runs)

### Data Files

- `extracted_data.json` - Extracted data from LLM outputs (used by analysis scripts)
- `reliability_analysis_results.json` - Reliability analysis results

### Other Directories

- **analysis/** - General analysis scripts (analyze_reliability.py)
- **analysis_strong_effect/** - Deprecated (empty after cleanup)
- **visualization/** - Visualization scripts
- **visualizations/** - Generated visualizations
- **database/** - Database files (formula_materials.json)
- **reporting/** - Reporting scripts and templates
- **claude/** - Claude-specific files

---

## popularity_bias/ - Popularity Bias Analysis Module

This directory contains the **popularity bias analysis module** using robust regression method.

### Structure

```
popularity_bias/
├── __init__.py           # Module interface (exports analyze_popularity_bias, RigorousBiasAnalyzer)
├── analysis/             # Analysis module
│   ├── __init__.py
│   └── robust_regression.py  # Main analysis logic (Partial Correlation + Robust Regression)
├── scripts/              # Legacy scripts (optional, for direct execution)
│   ├── analyze_rigorous_v2.py
│   ├── fetch_material_frequencies.py
│   ├── test_apis.py
│   ├── test_pubchem_two_step.py
│   ├── test_pubchem_urls.py
│   └── run_popularity_bias_analysis.py
├── data/                 # Input data files
│   ├── material_frequencies.json
│   ├── api_source_data.json
│   ├── api_cache.json
│   ├── formula_materials.json
│   ├── relative_frequencies.json
│   └── extracted_data.json
└── results/              # Output results
    ├── *_debiased_rigorous_v2.json
    ├── rigorous_analysis_v2_summary.json
    ├── rigorous_analysis_v2.log
    └── *.png
```

### Usage

#### Python API

```python
# Method 1: Convenience function
from popularity_bias import analyze_popularity_bias

results = analyze_popularity_bias()

# Method 2: Analyzer class
from popularity_bias import RigorousBiasAnalyzer

analyzer = RigorousBiasAnalyzer()
results = analyzer.run_rigorous_analysis()
```

#### Command Line Interface

```bash
# Using the unified API
python analyze.py popularity

# Or directly (legacy)
cd popularity_bias/scripts
python analyze_rigorous_v2.py
```

---

## anti_analysis/ - Anti-Formula Analysis Module

This directory contains the **anti-formula analysis module** for extracting arguments against specific formulas.

### Structure

```
anti_analysis/
├── __init__.py           # Module interface (exports extract_anti_arguments, AntiArgumentExtractor)
├── analysis/             # Analysis module
│   ├── __init__.py
│   └── extract_arguments.py   # Main extraction logic
├── scripts/              # Legacy scripts (optional, for direct execution)
│   └── extract_anti_arguments.py
└── results/              # Generated results
    ├── anti-Formula 4.md
    └── anti-Formula 5.md
```

### Usage

#### Python API

```python
# Method 1: Convenience function
from anti_analysis import extract_anti_arguments

arguments = extract_anti_arguments(formulas=[4, 5])

# Method 2: Extractor class
from anti_analysis import AntiArgumentExtractor

extractor = AntiArgumentExtractor()
arguments = extractor.extract(formulas=[5])
```

#### Command Line Interface

```bash
# Using the unified API
python analyze.py anti --formulas 4 5

# Or directly (legacy)
cd anti_analysis/scripts
python extract_anti_arguments.py
```

---

## Module System

### Unified API (analyze.py)

The `analyze.py` script provides a unified interface for running all analyses:

```bash
# Run popularity bias analysis
python analyze.py popularity

# Run anti-formula analysis
python analyze.py anti

# Run all analyses
python analyze.py all

# Show help
python analyze.py --help
```

### Python API

```python
from analyze import AnalysisPipeline

pipeline = AnalysisPipeline()

# Run specific analysis
popularity_results = pipeline.run_popularity_bias_analysis()
anti_results = pipeline.run_anti_formula_analysis(formulas=[4, 5])

# Run all analyses
all_results = pipeline.run_all()
```

### Module Exports

#### popularity_bias

- `analyze_popularity_bias()` - Convenience function
- `RigorousBiasAnalyzer` - Analyzer class

#### anti_analysis

- `extract_anti_arguments()` - Convenience function
- `AntiArgumentExtractor` - Extractor class

---

## Documentation

- **README.md** - This document, quick start guide
- **MODULE_USAGE.md** - Detailed module usage guide with examples
- **DIRECTORY_STRUCTURE.md** - This document

---

## Verification

To verify the new structure works correctly:

```bash
# Test popularity bias analysis module
python -c "from popularity_bias import analyze_popularity_bias, RigorousBiasAnalyzer; print('OK: popularity_bias module')"

# Test anti_analysis module
python -c "from anti_analysis import extract_anti_arguments, AntiArgumentExtractor; print('OK: anti_analysis module')"

# Test unified API
python analyze.py --help
```

All tests should pass without errors.