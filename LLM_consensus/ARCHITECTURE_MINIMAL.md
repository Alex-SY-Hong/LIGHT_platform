# LLM Consensus - Ultra-Compact Architecture (Single-Line)

## One-Line ASCII Flow

```
prompt.md → llm_concensus.py → API → (1)gpt-5.md (2)grok-4.md (3)claude.md (4)gemini.md → run_pipeline.py → [extract_data, analyze, visualize, report]
```

## Even More Compact

```
prompt → llm_concensus → API → [4×.md] → pipeline → [JSON/CSV, PNG, TEX]
```

## Vertical Flow (Compact)

```
prompt.md
    ↓
llm_concensus.py
    ↓
API (44 calls)
    ↓
[gpt-5, grok-4, claude, gemini].md
    ↓
run_pipeline.py
    ├→ analysis/ → JSON/CSV
    ├→ analyze → metrics.json
    ├→ visualize/ → 10 PNG
    └→ report/ → 2 TEX
```

## Minimal Version

```
.env + prompt → llm_concensus → [4×.md] → pipeline → [data, charts, reports]
```

## Pure ASCII (No Unicode)

```
prompt.md --> llm_concensus.py --> API --> (1)gpt-5.md (2)grok-4.md (3)claude.md (4)gemini.md --> run_pipeline.py --> [extract, analyze, visualize, report]
```

## Step-by-Step (One Line Each)

```
Step 1: llm_concensus.py → 44 API calls → 4 markdown files
Step 2: extract_data.py → parse markdown → JSON + CSV
Step 3: analyze_reliability.py → calculate CV, ICC, etc. → results.json
Step 4: visualization/*.py → generate 10 charts → *.png
Step 5: generate_tex*.py → generate reports → *.tex
```

## File Tree (Compact)

```
.
├── llm_concensus.py         # Data collection
├── [gpt-5, grok-4, claude, gemini].md    # Raw LLM output
├── run_pipeline.py          # Main pipeline
├── analysis/
│   ├── extract_data.py
│   └── analyze_reliability.py
├── visualization/
│   └── *.py (10 files)
├── reporting/
│   ├── generate_tex.py
│   └── generate_tex_en.py
├── extracted_data.json
├── reliability_results.json
├── visualizations/*.png
└── [*_CN, *_EN].tex
```
