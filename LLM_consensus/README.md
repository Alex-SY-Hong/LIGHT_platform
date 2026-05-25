# LLM Consensus Reliability Analysis

Statistical analysis of AI model reliability across multiple runs, evaluating stability and consistency in decision-making tasks.

## Project Overview

This project uses statistical methods to analyze the performance consistency of multiple AI models (GPT-5, Claude, Gemini, Grok) in materials science decision-making tasks, including:

- **Reliability Analysis**: Uses ICC (Intraclass Correlation Coefficient), CV (Coefficient of Variation), and other metrics to evaluate model consistency
- **Popularity Bias Analysis**: Analyzes whether models tend to select popular materials over optimal solutions
- **Anti-Formula Arguments**: Extracts arguments against specific formulations from model outputs

## Quick Start

### Environment Setup

```bash
# Using Poetry (recommended)
poetry install

# Or using pip
pip install -r requirements.txt
```

### Using Unified API Interface (Recommended)

The project provides a modularized API with both command-line and Python interfaces.

#### Command Line Interface

```bash
# Activate Poetry virtual environment
poetry shell

# Run popularity bias analysis
python analyze.py popularity

# Run anti-formula analysis
python analyze.py anti --formulas 4 5

# Run all analyses
python analyze.py all
```

#### Python API

```python
# Popularity bias analysis
from popularity_bias import analyze_popularity_bias

results = analyze_popularity_bias()
print(results['gpt-5']['optimal_formula'])

# Anti-formula analysis
from anti_analysis import extract_anti_arguments

arguments = extract_anti_arguments(formulas=[4, 5])
print(f"Formula 4: {len(arguments[4])} arguments")
print(f"Formula 5: {len(arguments[5])} arguments")

# Unified pipeline
from analyze import AnalysisPipeline

pipeline = AnalysisPipeline()
all_results = pipeline.run_all()
```

### Full Pipeline (Traditional)

```bash
# Activate virtual environment
poetry shell

# Run complete analysis pipeline
python run_pipeline.py
```

This will automatically execute:
1. Data extraction
2. Reliability analysis
3. Visualization generation
4. LaTeX report generation

### Running Individual Modules (Traditional)

```bash
# Run reliability analysis
python analyze_llm_reliability.py

# Run popularity bias analysis (direct script execution)
cd popularity_bias/scripts
python analyze_rigorous_v2.py

# Extract anti-formula arguments
cd ../../anti_analysis/scripts
python extract_anti_arguments.py

# Generate report (Chinese)
python generate_tex_report.py

# Generate report (English)
python generate_tex_report_en.py
```

## Project Structure

```
LLM_consensus/
├── analyze.py                  # Unified API interface (recommended)
├── llm_concensus.py            # LLM API calling script
├── run_pipeline.py             # One-click run script
├── analyze_llm_reliability.py  # Reliability analysis entry point
├── extract_data.py             # Data extraction tool
├── generate_tex_report.py      # Chinese LaTeX report generation
├── generate_tex_report_en.py   # English LaTeX report generation
│
├── popularity_bias/            # Popularity bias analysis module (recommended)
│   ├── __init__.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── robust_regression.py    # Main analysis logic
│   ├── scripts/               # Legacy scripts (optional)
│   │   ├── analyze_rigorous_v2.py
│   │   ├── fetch_material_frequencies.py
│   │   ├── test_apis.py
│   │   ├── test_pubchem_two_step.py
│   │   ├── test_pubchem_urls.py
│   │   └── run_popularity_bias_analysis.py
│   ├── data/                  # Input data
│   │   ├── material_frequencies.json
│   │   ├── relative_frequencies.json
│   │   ├── formula_materials.json
│   │   ├── extracted_data.json
│   │   └── api_*.json
│   └── results/               # Output results
│       ├── *_debiased_rigorous_v2.json
│       ├── rigorous_analysis_v2_summary.json
│       ├── rigorous_analysis_v2.log
│       └── *.png
│
├── anti_analysis/              # Anti-formula analysis module (recommended)
│   ├── __init__.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── extract_arguments.py    # Main extraction logic
│   ├── scripts/               # Legacy scripts (optional)
│   │   └── extract_anti_arguments.py
│   └── results/               # Output results
│       ├── anti-Formula 4.md
│       └── anti-Formula 5.md
│
├── analysis/                   # Data analysis module
│   ├── extract_data.py         # Data extraction
│   └── analyze_reliability.py  # Reliability analysis
├── analysis_strong_effect/     # Deprecated (old analysis scripts)
├── reporting/                  # Report generation module
│   ├── generate_tex.py         # Chinese report generation
│   └── generate_tex_en.py      # English report generation
├── visualization/              # Visualization module
│   ├── __init__.py
│   ├── load_data.py
│   ├── plot_overall.py
│   ├── plot_cv.py
│   ├── plot_icc.py
│   ├── plot_winner.py
│   ├── plot_ranking.py
│   ├── plot_entropy.py
│   ├── plot_detail.py
│   └── visualize_popularity_bias.py
├── database/                   # Data storage
│   └── formula_materials.json  # Formula-material mapping
├── claude/                     # Claude original response files
├── visualizations/             # Generated chart outputs
├── extracted_csv/              # Extracted CSV files
│
├── gpt-5.md                    # GPT-5 response file
├── grok-4.md                   # Grok-4 response file
├── claude-opus-4-5-20251101.md  # Claude Opus 4.5 response file
├── gemini-3-pro-preview.md     # Gemini 3 Pro response file
│
├── .env                        # Environment variable configuration
├── .env.example                # Environment variable example
├── pyproject.toml              # Poetry dependency management
├── requirements.txt            # Pip dependency list
├── DIRECTORY_STRUCTURE.md      # Directory structure documentation
├── MODULE_USAGE.md             # Module usage guide
└── README.md                   # Project documentation
```

### Modular Architecture

The project adopts a modular architecture with main analysis functions encapsulated as Python packages:

#### popularity_bias Module
- **Exports**: `analyze_popularity_bias()`, `RigorousBiasAnalyzer`
- **Methods**: Partial Correlation + Robust Regression (Huber + RANSAC)
- **Output**: JSON format debiasing results and statistical summary

#### anti_analysis Module
- **Exports**: `extract_anti_arguments()`, `AntiArgumentExtractor`
- **Function**: Extract arguments against specific formulas from LLM responses
- **Output**: Markdown format argument lists

#### analyze.py Unified Interface
- Provides command-line and Python usage methods
- Supports running individual analyses or all at once
- Automatic path detection and configuration

## Key Features

### 1. Reliability Analysis

Uses the following statistical metrics to evaluate model consistency:

- **ICC (Intraclass Correlation Coefficient)**: Intraclass correlation coefficient
  - ICC(3,k): Perfect agreement, fixed effects, average measurement
  - Reference: Shrout & Fleiss (1979)

- **CV (Coefficient of Variation)**: Coefficient of variation
  - CV < 10%: Excellent
  - 10% ≤ CV < 20%: Good
  - 20% ≤ CV < 30%: Fair
  - CV ≥ 30%: Poor

- **Entropy**: Evaluates uncertainty in formula selection

- **Winner Consistency**: Consistency in optimal formula selection

### 2. Popularity Bias Analysis

Uses scientifically rigorous methods to analyze whether models are influenced by material popularity:

**Methods (Robust Regression Module):**
- **Partial Correlation**: Control for confounding variables
- **Robust Regression**: Huber + RANSAC ensemble, outlier resistant
- **Permutation Test**: Non-parametric p-value testing, more rigorous than asymptotic methods

**Analysis Process:**
1. Calculate partial correlation coefficients for each dimension score vs. material popularity
2. Obtain p-values using permutation testing
3. Apply debiasing to dimensions exceeding thresholds
4. Recalculate total scores and determine optimal formula

**Parameters:**
- Debiasing threshold: `|ρ_partial| > 0.5 AND p < 0.10`
- Output: JSON format detailed results and visualization charts

### 3. Anti-Formula Arguments

Extracts arguments against specific formulas from model responses for manual review and integration.

## Output Files

Files generated after completion:

### Data Files
- `extracted_data.json` - Extracted raw data
- `reliability_analysis_results.json` - Reliability analysis results

### Popularity Bias Analysis Results
- `popularity_bias/results/` - Popularity bias analysis output directory
  - `{model}_debiased_rigorous_v2.json` - Debiasing results per model
  - `rigorous_analysis_v2_summary.json` - Analysis summary
  - `rigorous_analysis_v2.log` - Analysis log
  - `*.png` - Visualization charts

### Anti-Formula Analysis Results
- `anti_analysis/results/` - Anti-formula analysis output directory
  - `anti-Formula 4.md` - Arguments against Formula 4 (Markdown format)
  - `anti-Formula 5.md` - Arguments against Formula 5 (Markdown format)

### Visualizations
- `visualizations/overall_comparison.png` - Overall comparison radar chart
- `visualizations/cv_comparison.png` - CV comparison bar chart
- `visualizations/icc_heatmap.png` - ICC heatmap
- `visualizations/winner_consistency.png` - Winner consistency chart
- `visualizations/reliability_ranking.png` - Reliability ranking chart
- `visualizations/entropy_analysis.png` - Entropy analysis chart
- `visualizations/{model}_detail.png` - Model-specific detailed analysis charts

### Reports
- `LLM_Reliability_Report.tex/.pdf` - Chinese report
- `LLM_Reliability_Report_EN.tex/.pdf` - English report

## Technical Stack

- **Python 3.12**
- **Data Processing**: pandas, numpy
- **Statistical Analysis**: scipy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Report Generation**: LaTeX (xelatex)
- **Dependency Management**: Poetry, pip

## Known Issues

### GPT-5 Run 5 Data Extraction Problem

**Issue Description:**
In GPT-5's Run 5 response, the "Selected Formula" field lacks the formula number (format is "Gelatin_methacrylate (GelMA) & Polyethylene_glycol (PEG)" instead of "Formula 5 (GelMA & PEG)"), causing automatic extraction to fail.

**Solution:**
Since this issue only appeared once, it's recommended to manually set Run 5's Winner to 5 in `extracted_data.json` (to match the material in the response).

**Note:**
- For rigor, do not modify the original `.md` files
- If re-extracting data, this issue needs manual fixing again
- This issue does not affect other models

## Configuration

### Environment Variables (.env)

```
API_KEY=your_api_key_here
API_URL=your_api_url_here
```

### Modify LLM Model List

Edit the `models` variable in `llm_concensus.py`:

```python
models = [
    "gemini-3-pro-preview",
    "gpt-5",
    "grok-4",
    "claude-opus-4-5-20251101",
]
```

### Modify Number of Runs

Edit the `runs` variable in `llm_concensus.py`:

```python
runs = 11
```

## References

- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in assessing rater reliability. *Psychological Bulletin*, 86(2), 420-428.
- Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine*, 19(3), 342-349.

## License

MIT License

## Contributing

Issues and Pull Requests are welcome!

## Documentation

- **README.md** - This document, quick start guide
- **MODULE_USAGE.md** - Detailed module usage guide and API documentation
- **DIRECTORY_STRUCTURE.md** - Project directory structure documentation

## API Documentation

### popularity_bias Module

```python
from popularity_bias import analyze_popularity_bias, RigorousBiasAnalyzer

# Method 1: Convenience function
results = analyze_popularity_bias()

# Method 2: Analyzer class
analyzer = RigorousBiasAnalyzer()
results = analyzer.run_rigorous_analysis()

# Access results
for model, model_results in results.items():
    optimal = model_results['optimal_formula']
    print(f"{model}: Formula {optimal['formula_id']} - Score: {optimal['total_score']:.2f}")
```

### anti_analysis Module

```python
from anti_analysis import extract_anti_arguments, AntiArgumentExtractor

# Method 1: Convenience function
arguments = extract_anti_arguments(formulas=[4, 5])

# Method 2: Extractor class
extractor = AntiArgumentExtractor()
arguments = extractor.extract(formulas=[5])

# Access results
for formula_num, args in arguments.items():
    print(f"Formula {formula_num}: {len(args)} arguments")
```

### analyze.py Unified Interface

```python
from analyze import AnalysisPipeline

pipeline = AnalysisPipeline()

# Run specific analysis
pop_results = pipeline.run_popularity_bias_analysis()
anti_results = pipeline.run_anti_formula_analysis(formulas=[4, 5])

# Run all analyses
all_results = pipeline.run_all()

# Custom configuration
pop_results = pipeline.run_popularity_bias_analysis(
    data_dir="./custom/data",
    results_dir="./custom/results",
    configure_logging=False
)
```

## Command Line Help

```bash
# View all available commands
python analyze.py --help

# View specific command help
python analyze.py popularity --help
python analyze.py anti --help
python analyze.py all --help
```

## Typical Workflows

### Scenario 1: Quick popularity bias analysis

```bash
poetry shell
python analyze.py popularity
```

### Scenario 2: Extract arguments for specific formula

```bash
poetry shell
python analyze.py anti --formulas 5
```

### Scenario 3: Complete analysis workflow

```bash
poetry shell
python analyze.py all
python generate_tex_report.py
```

### Scenario 4: Using modules in Python scripts

```python
from popularity_bias import analyze_popularity_bias
from anti_analysis import extract_anti_arguments
from analyze import AnalysisPipeline

# Run analyses
popularity_results = analyze_popularity_bias()
anti_arguments = extract_anti_arguments(formulas=[4, 5])

# Or use unified interface
pipeline = AnalysisPipeline()
all_results = pipeline.run_all()

# Custom post-processing
for model, result in popularity_results.items():
    formula_id = result['optimal_formula']['formula_id']
    score = result['optimal_formula']['total_score']
    print(f"{model} recommends Formula {formula_id} with score {score:.2f}")
```