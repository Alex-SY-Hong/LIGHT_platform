# LLM Consensus Reliability Analysis - Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LLM RELIABILITY ANALYSIS PLATFORM                        │
│                         Project Architecture                                │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                            UPSTREAM LAYER                                     │
│                                                                              │
│  ┌──────────────────────┐         ┌──────────────────────┐                  │
│  │     .env file        │         │    prompt.md         │                  │
│  │  ┌────────────────┐  │         │  ┌────────────────┐  │                  │
│  │  │ API_KEY        │  │         │  │ System Role    │  │                  │
│  │  │ API_URL        │  │         │  │ Context        │  │                  │
│  │  │ (OpenAI        │  │         │  │ 10 Formulas    │  │                  │
│  │  │  compatible)   │  │         │  │ Scoring        │  │                  │
│  │  └────────────────┘  │         │  │ Protocol       │  │                  │
│  └──────────┬───────────┘         └──────────┬─────────┘                  │
│             │                                │                            │
│             └────────────┬───────────────────┘                            │
│                          │                                                │
│                          ▼                                                │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              llm_concensus.py (Data Collection)                    │    │
│  │                                                                     │    │
│  │  • Configure 4 models: gemini-3-pro, gpt-5, grok-4, claude-opus   │    │
│  │  • Set 11 runs per model (temperature=0.0 for consistency)        │    │
│  │  • Loop through models × runs                                     │    │
│  │  • Call OpenAI-compatible API                                     │    │
│  │  • Append responses to {model_name}.md files                      │    │
│  │  • Sleep 5s between requests                                      │    │
│  └───────────────────────────────┬───────────────────────────────────┘    │
│                                  │                                         │
│                                  ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              🌐 External LLM API (OpenAI-compatible)              │    │
│  │                                                                 │    │
│  │  API Request:                                                    │    │
│  │  • model: {model_name}                                           │    │
│  │  • messages: [{role: "user", content: prompt}]                   │    │
│  │  • temperature: 0.0 (deterministic)                              │    │
│  │  • stream: False                                                 │    │
│  │                                                                 │    │
│  │  API Response:                                                   │    │
│  │  • CSV scores table (10 formulas × 7 metrics)                   │    │
│  │  • Winner selection with rationale                               │    │
│  │  • Detailed analysis (mechanical fit, swelling, etc.)            │    │
│  │  • Rejected candidates with reasons                              │    │
│  └───────────────────────────────┬───────────────────────────────────┘    │
│                                  │                                         │
└──────────────────────────────────┼─────────────────────────────────────────┘
                                   │
                                   ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   gpt-5.md   │  │  grok-4.md   │  │ claude*.md   │  │ gemini*.md   │       │
│  │  (11 runs)   │  │  (11 runs)   │  │  (11 runs)   │  │  (11 runs)   │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                 │                 │               │
│         └─────────────────┴─────────────────┴─────────────────┘               │
│                                    │                                          │
│                                    ▼                                          │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                            PROCESSING LAYER                                   │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │                        🔄 run_pipeline.py                              │   │
│  │                      (One-click Orchestration)                         │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                          │
│           ┌────────────────────────┼────────────────────────┐                 │
│           │                        │                        │                 │
│           ▼                        ▼                        ▼                 │
│  ┌────────────────┐    ┌─────────────────┐    ┌──────────────────┐            │
│  │   analysis/    │    │   reporting/    │    │ visualization/   │            │
│  │   ┌────────┐   │    │   ┌──────────┐  │    │   ┌───────────┐  │            │
│  │   │extract │   │    │   │generate  │  │    │   │ visualize │  │            │
│  │   │_data   │   │    │   │_tex (CN) │  │    │   │_utils.py  │  │            │
│  │   └────┬───┘   │    │   └────┬─────┘  │    │   └─────┬─────┘  │            │
│  │        │       │    │        │        │    │         │        │            │
│  │   ┌────▼─────┐ │    │   ┌────▼─────┐  │    │   ┌─────▼─────┐  │            │
│  │   │analyze   │ │    │   │generate  │  │    │   │  Multiple │  │            │
│  │   │_reliable │ │    │   │_tex_en   │  │    │   │  plots    │  │            │
│  │   └────┬─────┘ │    │   └──────────┘  │    │   │  (CV,ICC, │  │            │
│  └────────┼───────┘    └─────────────────┘    │   │   etc.)   │  │            │
│           │                                   │   └───────────┘  │            │
│           └───────────────────────────────────┴──────────────────┘            │
│                                   │                                           │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                             OUTPUT LAYER                                      │
│                                                                              │
│  📊 DATA FILES                    📈 VISUALIZATIONS      📄 REPORTS        │
│  ┌────────────────────┐         ┌───────────────┐      ┌───────────────┐   │
│  │ extracted_data.json│         │ *.png (10 files)│     │ *.tex files   │   │
│  │ extracted_csv/     │         │ • cv_comparison│     │ (CN & EN)     │   │
│  │   ├── all_models.csv│         │ • icc_heatmap │      │               │   │
│  │   └── {model}.csv  │         │ • ranking     │      │ Compile with: │   │
│  └────────────────────┘         │ • consistency │      │ xelatex ×2    │   │
│  ┌────────────────────┐         │ • entropy    │      └───────────────┘   │
│  │ reliability_       │         │ • margin     │                            │
│  │ analysis_results   │         └───────────────┘                            │
│  │    .json           │                                                      │
│  └────────────────────┘         ┌───────────────┐                            │
│                                 │ visualizations/│                           │
│                                 │ directory      │                           │
│                                 └───────────────┘                           │
└───────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         CONFIGURATION & CONTEXT                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  .venv/          │    │  claude/         │    │  README.md       │        │
│  │  (Python env)    │    │  • CONTEXT.json │    │  • Quick start   │        │
│  │  • pyproject.toml│    │  • 关键指令.md  │    │  • Usage guide   │        │
│  │                  │    │  • README.md    │    │                  │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW SUMMARY                                     │
│                                                                             │
│  0️⃣  DATA COLLECTION (Upstream)                                            │
│      llm_concensus.py → Call LLM APIs → Generate 4× markdown files          │
│      Configuration: .env + prompt.md                                        │
│      Output: gpt-5.md, grok-4.md, claude*.md, gemini*.md (each 11 runs)    │
│                                                                             │
│  1️⃣  DATA EXTRACTION                                                       │
│      extract_data.py → Parse markdown files → Generate JSON/CSV             │
│                                                                             │
│  2️⃣  RELIABILITY ANALYSIS                                                  │
│      analyze_reliability.py → Calculate metrics → Generate results JSON     │
│      Metrics: CV, ICC, Consistency, Entropy, Winner Margin                  │
│                                                                             │
│  3️⃣  VISUALIZATION                                                         │
│      visualization/ package → Generate 10 PNG charts                        │
│                                                                             │
│  4️⃣  REPORT GENERATION                                                     │
│      generate_tex.py → Chinese LaTeX report                                 │
│      generate_tex_en.py → English LaTeX report                              │
│                                                                             │
│  ⚡  QUICK START                                                            │
│      # Full workflow (data collection + analysis)                           │
│      python llm_concensus.py && python run_pipeline.py                      │
│      # Analysis only (if data already collected)                            │
│      python run_pipeline.py                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    STATISTICAL METRICS COMPUTED                              │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │ Coefficient of      │  │ Intraclass          │  │ Decision            │  │
│  │ Variation (CV)      │  │ Correlation         │  │ Consistency Rate    │  │
│  │                    │  │ Coefficient (ICC)   │  │                     │  │
│  │ Measures score      │  │                     │  │ % runs selecting    │  │
│  │ dispersion         │  │ ANOVA-based         │  │ same winner         │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                                                             │
│  ┌─────────────────────┐  ┌─────────────────────┐                            │
│  │ Information         │  │ Winner Margin       │                            │
│  │ Entropy             │  │ (Gap between 1st    │                            │
│  │                     │  │ and 2nd place)      │                            │
│  │ Decision            │  │                     │                            │
│  │ uncertainty         │  │ Cascading           │                            │
│  │                     │  │ divergence risk     │                            │
│  └─────────────────────┘  └─────────────────────┘                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          MODELS ANALYZED                                    │
│                                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │    GPT-5     │  │    Grok-4    │  │ Claude Opus  │  │  Gemini 3    │    │
│  │              │  │              │  │    4.5       │  │    Pro       │    │
│  │ 90.9%        │  │  63.6%       │  │  63.6%       │  │  100.0%      │    │
│  │ consistency  │  │  consistency │  │  consistency │  │  consistency │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                             │
│              Winner: Gemini 3 Pro (100% consistency, ICC=0.9056)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Reference

### Data Collection (Upstream)

| Script | Purpose | Config |
|--------|---------|--------|
| **llm_concensus.py** | Collect LLM responses via API | `.env` + `prompt.md` |
| **.env** | API credentials | `API_KEY`, `API_URL` |
| **prompt.md** | Evaluation task prompt | Role, context, 10 formulas, scoring protocol |

**Data Collection Workflow:**
```bash
# Configure API credentials in .env
API_KEY=your_key_here
API_URL=https://your-api-endpoint

# Run data collection (takes ~15-20 minutes with 5s delay)
python llm_concensus.py

# Output: 4 markdown files with 11 runs each
# Total API calls: 4 models × 11 runs = 44 requests
```

### Core Modules

| Module | File | Purpose |
|--------|------|---------|
| **Data Extraction** | `analysis/extract_data.py` | Parse markdown → JSON/CSV |
| **Reliability Analysis** | `analysis/analyze_reliability.py` | Calculate statistical metrics |
| **Chinese Report** | `reporting/generate_tex.py` | Generate LaTeX report (中文) |
| **English Report** | `reporting/generate_tex_en.py` | Generate LaTeX report (英文) |
| **Visualization** | `visualization/*.py` | Generate all charts |
| **Pipeline** | `run_pipeline.py` | One-click full workflow |

### Key Commands

```bash
# Activate environment
.venv\Scripts\activate

# Step 1: Data Collection (run first to collect LLM responses)
python llm_concensus.py
# Output: gpt-5.md, grok-4.md, claude-opus-4-5-20251101.md, gemini-3-pro-preview.md

# Step 2: Run full pipeline (analysis + visualization + reports)
python run_pipeline.py

# Individual steps (if needed)
python -c "from analysis import extract_data; extract_data.main()"
python -c "from analysis import analyze_reliability; analyze_reliability.main()"
python example_visualization.py
python -c "from reporting import generate_tex, generate_tex_en; ..."
```

### Output Files

**From Data Collection (llm_concensus.py):**
- `gpt-5.md` - GPT-5 responses (11 runs)
- `grok-4.md` - Grok-4 responses (11 runs)
- `claude-opus-4-5-20251101.md` - Claude Opus 4.5 responses (11 runs)
- `gemini-3-pro-preview.md` - Gemini 3 Pro responses (11 runs)

**From Analysis Pipeline (run_pipeline.py):**
- **Data**: `extracted_data.json`, `extracted_csv/*.csv`
- **Results**: `reliability_analysis_results.json`
- **Charts**: `visualizations/*.png` (10 files)
- **Reports**: `LLM_Reliability_Report.tex`, `LLM_Reliability_Report_EN.tex`
