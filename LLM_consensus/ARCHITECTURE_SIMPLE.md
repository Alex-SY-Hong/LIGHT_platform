# LLM Consensus Reliability Analysis - Simplified Architecture

## One-Line Flow

```
.env + prompt.md → llm_concensus.py → LLM API → [gpt-5, grok-4, claude, gemini].md → run_pipeline.py → [JSON/CSV, .tex, .png]
```

## ASCII Flow Diagram

```
                    ┌────────────┐
                    │    api     │ 
                    │  endpoint  │
                    │   prompt   │
                    └─────┬──────┘
                          │
                          │
                    ┌─────▼──────┐
                    │  LLM API   │
                    │(OpenAI)    │
                    └─────┬──────┘
                          │
              ┌───────────┼───────────┼─────────┐
              │           │           │         │
         ┌────▼───┐  ┌────▼───┐ ┌───▼────┐ ┌────▼───┐
         │ gpt-5  │  │grok-4  │ │claude  │ │gemini  │
         └────┬───┘  └────┬───┘ └────┬───┘ └────┬───┘
              │           │          │          │
              └───────────┴──────────┴──────────┘
                          │
                    ┌─────▼──────┐
                    │run_pipeline│
                    └─────┬──────┘
                          │
        ┌─────────────────┼────────────────┐
        │                 │                │
   ┌────▼─────┐      ┌────▼─────┐     ┌────▼──────┐
   │ analysis │      │ reporting│     │visualize  │
   │          │      │          │     │           │
   │extract   │      │generate  │     │  10 PNG   │
   │analyze   │      │_tex      │     │  charts   │
   └────┬─────┘      │(CN+EN)   │     └───────────┘
        │            └────┬─────┘
        │                │
        ▼                ▼
   ┌─────────┐     ┌─────────┐
   │JSON/CSV │     │  .tex   │
   └─────────┘     └─────────┘
```

## Ultra-Compact Version

```
prompt → llm_concensus.py → API → (1)gpt-5.md (2)grok-4.md (3)claude.md (4)gemini.md → run_pipeline.py → [data, charts, reports]
```

## Data Flow Summary

```
Step 0: prompt.md + .env
             ↓
Step 1: llm_concensus.py → 44 API calls (4 models × 11 runs)
             ↓
Step 2: 4 markdown files (raw LLM responses)
             ↓
Step 3: run_pipeline.py
        ├→ analysis/ → extracted_data.json + CSV
        ├→ analyze_reliability → results.json (CV, ICC, etc.)
        ├→ visualization/ → 10 PNG charts
        └→ reporting/ → 2 LaTeX reports (CN + EN)
```

## Module Dependency Graph

```
llm_concensus.py (standalone)
    ↓
{model_name}.md (4 files)
    ↓
run_pipeline.py
    ├→ analysis/__init__.py
    │   ├→ extract_data.py
    │   └→ analyze_reliability.py
    ├→ visualization/__init__.py
    │   └→ *.py (10 chart modules)
    └── reporting/__init__.py
        ├→ generate_tex.py
        └── generate_tex_en.py
```

## Core Scripts (5 files)

```
llm_concensus.py     → Data collection
run_pipeline.py      → Main orchestrator
extract_data.py      → Parse markdown
analyze_reliability.py → Calculate metrics
generate_tex*.py     → Generate reports
```

## Input → Output Mapping

```
INPUT                    PROCESS                 OUTPUT
────────────────────────────────────────────────────────────
.env + prompt.md  →  llm_concensus.py    →  [gpt-5, grok-4, claude, gemini].md
*.md (4 files)   →  extract_data.py     →  extracted_data.json + CSV
*.json           →  analyze_reliability → reliability_results.json
*.json           →  visualization/      →  10 PNG charts
*.json           →  reporting/         →  2 LaTeX reports
```
