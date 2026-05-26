# LLM Consensus - Semantic Architecture

## One-Line Semantic Flow

```
{upstream: {config, prompt} → call_llm → {raw_data}} → {process: raw_data → analyze} → {output: {report, chart, data}}
```

## ASCII Semantic Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│  UPSTREAM                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐          │
│  │  {config}    │  →   │  call_llm    │  →   │  {raw_data}  │          │
│  │              │      │              │      │              │          │
│  │  .env        │      │  API calls   │      │  4× .md      │          │
│  │  prompt.md   │      │              │      │  files       │          │
│  └──────────────┘      └──────────────┘      └──────────────┘          │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  PROCESS                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐          │
│  │  {raw_data}  │  →   │  {extract}   │  →   │  {analyze}   │          │
│  │              │      │              │      │              │          │
│  │  4× .md      │      │  parse data  │      │  statistics  │          │
│  │  files       │      │  normalize   │      │  metrics     │          │
│  └──────────────┘      └──────────────┘      └──────────────┘          │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                                │
│  ┌───────────────────────────────────────────────────────────────────┐ │
│  │                        {analyze_result}                           │ │
│  │                                                                   │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │ │
│  │  │  {report}   │  │  {figure}   │  │   {data}    │                │ │
│  │  │             │  │             │  │             │                │ │
│  │  │  LaTeX      │  │  PNG        │  │  JSON/CSV   │                │ │
│  │  │  CN + EN    │  │  10 files   │  │  files      │                │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Phase Abstraction

```
┌─ PHASE 0: CONFIGURE ─────────────────────────────────────────────────────┐
│  {input: user_config} → {action: setup} → {output: api_ready}               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─ PHASE 1: COLLECT ──────────────────────────────────────────────────────────┐
│  {input: api_ready} → {action: query_llm} → {output: raw_responses}          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─ PHASE 2: PROCESS ──────────────────────────────────────────────────────────┐
│  {input: raw_responses} → {action: extract_analyze} → {output: metrics}      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─ PHASE 3: OUTPUT ───────────────────────────────────────────────────────────┐
│  {input: metrics} → {action: generate} → {output: reports_charts_data}     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Type-Level Flow

```
[Config]  -->  [API]  -->  [Markdown]×4  -->  [Structured]  -->  [Metrics]  -->  [Artifacts]
    ↓           ↓           ↓                ↓              ↓             ↓
  String    Response     TextFiles      JSON/CSV        FloatDict     Files
```

## Minimal Semantic

```
{configure → collect → process → output}
```

## Data Transformation Pipeline

```
Text (LLM output)
    ↓ extract
Structured (JSON/CSV)
    ↓ analyze
Metrics (Float array)
    ↓ visualize
Charts (PNG)
    ↓ report
Documents (PDF)
```

## Module-Level Abstraction

```
┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│config│ → │collect│ → │extract│ → │analyze│ → │render│ → │output│
└──────┘   └──────┘   └──────┘   └──────┘   └──────┘   └──────┘
  setup      API        parse       stats       LaTeX      PDF
              query                  calc                    PNG
```

## Output Categorization

```
┌─ REPORTS ────────────────────────────────────────────────────────────────┐
│  {type: document, format: PDF×2, language: [CN, EN], engine: LaTeX}      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ CHARTS ─────────────────────────────────────────────────────────────────────┐
│  {type: visualization, format: PNG×10, content: [CV, ICC, ranking, ...]}   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─ DATA ────────────────────────────────────────────────────────────────────────┐
│  {type: dataset, format: [JSON, CSV], records: 440, models: 4, runs: 11}    │
└─────────────────────────────────────────────────────────────────────────────┘
```
