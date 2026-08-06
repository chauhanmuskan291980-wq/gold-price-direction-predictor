# Signal Validation Gate
# Software Design Document (SDD)

> **Project Type:** Statistical Validation Framework for Quantitative Trading Strategies  
> **Language:** Python 3.10+  
> **Architecture:** Modular Layered Architecture (Pipeline Pattern + Single Responsibility Principle)  
> **Author:** Muskan Chauhan

---

# Table of Contents

1. Introduction
2. Problem Statement
3. System Objectives
4. High-Level Architecture
5. Validation Pipeline
6. Software Design Pattern
7. Project Structure
8. Module Responsibilities
9. Data Flow
10. Validation Algorithms
11. Statistical Components
12. Validation Decision Framework
13. Testing Strategy
14. CLI Workflow
15. Design Principles
16. Future Improvements

---

# 1. Introduction

The **Signal Validation Gate** is a statistical validation framework designed to determine whether a quantitative trading strategy demonstrates a genuine statistical edge **before** it is used for machine learning model training.

Instead of relying on a single performance metric such as Profit Factor or Total Return, the framework combines multiple independent statistical validation techniques to measure robustness, consistency, and statistical significance.

The framework is designed to answer one question:

> **"Does this strategy truly contain predictive information, or is the observed performance likely caused by randomness?"**

---

# 2. Problem Statement

Many trading strategies appear profitable because of:

- Random chance
- Curve fitting
- Data snooping
- Multiple hypothesis testing
- Overfitting
- Small sample sizes

Training an ML model on these strategies often results in poor real-world performance.

The Signal Validation Gate ensures that only statistically validated strategies are passed to downstream ML pipelines.

---

# 3. System Objectives

The framework is designed to:

- Support multiple trading CSV formats
- Normalize heterogeneous datasets
- Perform chronological walk-forward evaluation
- Compute robust trading statistics
- Estimate confidence using bootstrap resampling
- Correct for multiple testing using permutation null distributions
- Produce reproducible validation reports
- Return a deterministic final verdict

---

# 4. High-Level Architecture

```mermaid
flowchart TD

A[Trading CSV] --> B[Input Loader]
B --> C[Strategy Normalizer]

C --> D[Walk Forward Analysis]
C --> E[Trade Metrics]
C --> F[Bootstrap]
C --> G[Selection-Adjusted Null Test]

D --> H[Verdict Evaluation]
E --> H
F --> H
G --> H

H --> I[Temporary Validation Report]
I --> J[CLI Output]
```

---

# 5. Complete Validation Pipeline

```mermaid
flowchart LR

A[CSV File]
--> B[Load CSV]

B --> C[Detect Input Type]

C --> D[Normalize Strategy]

D --> E[Extract Trade Returns]

E --> F[Walk Forward]

E --> G[Trade Metrics]

E --> H[Bootstrap]

E --> I[Permutation Null]

F --> J[Verdict Engine]
G --> J
H --> J
I --> J

J --> K[Validation Report]
```

---

# 6. Software Design Pattern

The project follows a **Layered Pipeline Architecture**.

## Layer 1 — Data Access

Responsible for reading strategy files.

```
input_loader.py
```

---

## Layer 2 — Data Transformation

Converts every supported strategy format into one standardized representation.

```
normalization.py
```

---

## Layer 3 — Statistical Analysis

Independent statistical modules.

```
windows.py
metrics.py
bootstrap.py
nulls.py
```

Each module performs exactly one task.

---

## Layer 4 — Decision Layer

```
verdict.py
```

Consumes statistical evidence and generates the final decision.

---

## Layer 5 — Orchestration Layer

```
validation.py
```

Coordinates every component.

No statistical logic lives here.

---

## Layer 6 — Presentation Layer

```
cli.py
```

Responsible only for displaying results.

---

# 7. Project Structure

```text
gate/
│
├── __init__.py
├── __main__.py
├── cli.py
├── validation.py
├── schemas.py
│
├── input_loader.py
├── normalization.py
├── metrics.py
├── windows.py
├── bootstrap.py
├── nulls.py
├── verdict.py
```

---

# 8. Module Responsibilities

## input_loader.py

**Purpose**

Reads strategy CSV files.

### Responsibilities

- Detect input format
- Validate required columns
- Return DataFrame
- Identify InputType

---

## normalization.py

**Purpose**

Transforms all supported strategy formats into a common representation.

### Output

```
NormalizedStrategy
```

Contains:

- trade_returns
- trade_count
- return_unit
- input_type

---

## windows.py

**Purpose**

Performs chronological walk-forward validation.

Produces:

```
WalkForwardSummary
```

Metrics:

- Window count
- Median expectancy
- Worst expectancy
- Positive window rate
- Expectancy IQR

---

## metrics.py

Computes observed trading statistics.

Produces:

```
TradeMetricsSummary
```

Includes:

- Expectancy
- Profit Factor
- Win Rate
- Gross Profit
- Gross Loss
- Losing Streak
- Winner Concentration

---

## bootstrap.py

Performs bootstrap resampling.

Produces:

```
BootstrapSummary
```

Contains:

- Observed expectancy
- Lower confidence bound
- Profit factor lower bound
- Iterations
- Seed

---

## nulls.py

Performs selection-adjusted permutation testing.

Produces:

```
NullSummary
```

Contains:

- Adjusted p-value
- Null threshold
- Observed statistic

---

## verdict.py

Combines every statistical result into one decision.

Produces:

```
VerdictAssessment
```

Possible outputs:

- EDGE
- NO-EDGE
- INSUFFICIENT DATA

---

## validation.py

Acts as the orchestration layer.

Pipeline:

```
Loader
    ↓
Normalizer
    ↓
Walk Forward
    ↓
Metrics
    ↓
Bootstrap
    ↓
Null Test
    ↓
Verdict
    ↓
Validation Report
```

---

## cli.py

Presentation layer.

Displays:

- Walk Forward
- Metrics
- Bootstrap
- Null Test
- Verdict

---

# Sequence Diagram

---
```mermaid
sequenceDiagram

participant User
participant CLI
participant Validation
participant Loader
participant Normalizer
participant Metrics
participant Windows
participant Bootstrap
participant NullTest
participant Verdict

User->>CLI: python -m gate validate

CLI->>Validation: validate()

Validation->>Loader: load_strategy_csv()

Loader-->>Validation: DataFrame

Validation->>Normalizer: normalize_strategy()

Normalizer-->>Validation: NormalizedStrategy

Validation->>Metrics: calculate_trade_metrics()

Metrics-->>Validation: TradeMetricsSummary

Validation->>Windows: calculate_walk_forward_summary()

Windows-->>Validation: WalkForwardSummary

Validation->>Bootstrap: run_bootstrap()

Bootstrap-->>Validation: BootstrapSummary

Validation->>NullTest: run_selection_adjusted_null()

NullTest-->>Validation: NullSummary

Validation->>Verdict: evaluate_verdict()

Verdict-->>Validation: VerdictAssessment

Validation-->>CLI: TemporaryValidationReport

CLI-->>User: Display Validation Report
```
---

# Module Dependency Diagram

---
```mermaid
graph TD

CLI[cli.py]

Validation[validation.py]

Loader[input_loader.py]
Normalizer[normalization.py]
Metrics[metrics.py]
Windows[windows.py]
Bootstrap[bootstrap.py]
Nulls[nulls.py]
Verdict[verdict.py]
Schemas[schemas.py]

CLI --> Validation

Validation --> Loader
Validation --> Normalizer
Validation --> Metrics
Validation --> Windows
Validation --> Bootstrap
Validation --> Nulls
Validation --> Verdict

Verdict --> Schemas
Loader --> Schemas
Normalizer --> Schemas
Metrics --> Schemas
Windows --> Schemas
Bootstrap --> Schemas
Nulls --> Schemas
```
---

# Decision Flowchart
---
```mermaid
flowchart TD

Start([Start])

Start --> A{Enough Trades?}

A -- No --> ID[INSUFFICIENT DATA]

A -- Yes --> B{Enough Walk Forward Windows?}

B -- No --> ID

B -- Yes --> C{Positive Expectancy?}

C -- No --> NE[NO-EDGE]

C -- Yes --> D{Bootstrap Lower Bound > 0?}

D -- No --> NE

D -- Yes --> E{Adjusted P-value < 0.05?}

E -- No --> NE

E -- Yes --> F{Winner Concentration OK?}

F -- No --> NE

F -- Yes --> G{Losing Streak OK?}

G -- No --> NE

G -- Yes --> EDGE[EDGE]
```
---

# Class Diagram
---
```mermaid
classDiagram

class TemporaryValidationReport{
+Path source_path
+InputType input_type
+int trade_count
+WalkForwardSummary walk_forward
+TradeMetricsSummary metrics
+BootstrapSummary bootstrap
+NullSummary null_test
+VerdictAssessment verdict_assessment
}

class WalkForwardSummary

class TradeMetricsSummary

class BootstrapSummary

class NullSummary

class VerdictAssessment

TemporaryValidationReport --> WalkForwardSummary
TemporaryValidationReport --> TradeMetricsSummary
TemporaryValidationReport --> BootstrapSummary
TemporaryValidationReport --> NullSummary
TemporaryValidationReport --> VerdictAssessment
```
---
# 9. Data Flow

```text
CSV File
    │
    ▼
load_strategy_csv()
    │
    ▼
normalize_strategy()
    │
    ▼
trade_returns
    │
    ├──────────────┐
    ▼              ▼
Trade Metrics   Walk Forward
    │              │
    └──────┬───────┘
           ▼
     Bootstrap
           │
           ▼
 Selection Null Test
           │
           ▼
 evaluate_verdict()
           │
           ▼
TemporaryValidationReport
           │
           ▼
CLI Output
```

---

# 10. Validation Algorithms

The validation framework combines multiple independent statistical analyses.

| Stage | Purpose |
|---------|----------|
| Walk Forward | Stability through time |
| Trade Metrics | Observed performance |
| Bootstrap | Confidence estimation |
| Null Test | Statistical significance |
| Verdict | Rule-based decision |

---

# 11. Statistical Components

## Walk Forward

Evaluates chronological robustness.

Measures:

- Median Expectancy
- Worst Window
- Positive Window Rate

---

## Trade Metrics

Measures actual trading performance.

Includes:

- Expectancy
- Profit Factor
- Win Rate
- Losing Streak
- Winner Concentration

---

## Bootstrap

Uses random resampling to estimate confidence.

Answers:

> "How much could these statistics change due to random sampling?"

---

## Selection-Adjusted Null

Uses sign-permutation testing.

Answers:

> "If many strategy configurations were tested, is this strategy still statistically significant?"

---

## Verdict Policy

Combines all evidence.

Requirements include:

- Minimum trade count
- Minimum walk-forward windows
- Positive expectancy
- Positive bootstrap lower bounds
- Significant adjusted p-value
- Acceptable winner concentration
- Limited losing streak

---

# 12. Validation Decision Framework

```text
                    Statistical Evidence
                             │
                             ▼
                 ┌──────────────────────┐
                 │  Verdict Assessment  │
                 └──────────┬───────────┘
                            │
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
 INSUFFICIENT DATA       NO-EDGE              EDGE
```

---

# 13. Testing Strategy

Each module is verified independently.

## Static Analysis

```bash
ruff check gate
mypy gate
```

---

## Manual Validation

```bash
python -m gate validate ...
```

---

## Programmatic Testing

```python
report = validate(...)
print(report.to_dict())
```

---

## Future

- pytest
- integration tests
- benchmark tests
- property-based testing

---

# 14. CLI Workflow

```text
User
 │
 ▼
python -m gate validate
 │
 ▼
Validation Pipeline
 │
 ▼
Statistical Analysis
 │
 ▼
Final Report
```

---

# 15. Design Principles

The project follows several software engineering principles.

## Single Responsibility Principle (SRP)

Each module has exactly one responsibility.

---

## Separation of Concerns

- Loading
- Normalization
- Statistics
- Decision
- Presentation

remain completely independent.

---

## Pipeline Pattern

Each stage consumes the previous stage's output.

```
Input

↓

Transform

↓

Analyze

↓

Decide

↓

Report
```

---

## Immutable Data Objects

All summaries use frozen dataclasses.

Benefits:

- Thread-safe
- Predictable
- Easy testing

---

## Type Safety

The project uses:

- Type hints
- mypy
- dataclasses
- enums

to minimize runtime errors.

---

# 🚀 Getting Started

This guide explains how to set up and run the Signal Validation Gate from scratch.

---

# System Requirements

- Python 3.10+
- Git
- pip
- Virtual Environment (recommended)

---

# 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/gold-price-direction-predictor.git

cd gold-price-direction-predictor
```

---

# 2. Create a Virtual Environment

Windows

```bash
python -m venv .venv
```

Linux / macOS

```bash
python3 -m venv .venv
```

---

# 3. Activate the Virtual Environment

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate again.

### Windows CMD

```cmd
.venv\Scripts\activate.bat
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

# 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 5. Verify Installation

```bash
ruff check gate

mypy gate
```

Expected output

```text
All checks passed!

Success: no issues found
```

---

# Project Structure

```
gold-price-direction-predictor/

│
├── gate/
│   ├── cli.py
│   ├── validation.py
│   ├── verdict.py
│   ├── bootstrap.py
│   ├── metrics.py
│   ├── windows.py
│   ├── nulls.py
│   ├── normalization.py
│   ├── input_loader.py
│   └── schemas.py
│
├── tests/
│   └── fixtures/
│       └── sample_closed_trades.csv
│
└── README.md
```

---

# Running the Validation Framework

The framework is executed through the CLI.

Example:

```bash
python -m gate validate \
    --trades tests/fixtures/sample_closed_trades.csv
```

---

## Full Validation

```bash
python -m gate validate \
    --trades tests/fixtures/sample_closed_trades.csv \
    --window-size 50 \
    --step-size 50 \
    --configs-tried 10 \
    --null-iterations 5000
```

The framework automatically performs:

```
Load CSV
      ↓
Normalize Strategy
      ↓
Observed Metrics
      ↓
Walk-Forward Analysis
      ↓
Bootstrap Analysis
      ↓
Selection-Adjusted Null Test
      ↓
Final Verdict
```

---

# Input CSV Format

The validator currently accepts closed trade history.

Example:

| open_time | close_time | side | return_R |
|-----------|------------|------|----------|
| 2024-01-01 | 2024-01-02 | LONG | 1.25 |
| 2024-01-02 | 2024-01-03 | SHORT | -0.50 |
| 2024-01-03 | 2024-01-04 | LONG | 0.80 |

Required columns

```
open_time
close_time
side
return_R
```

---

# Using Your Own Strategy

Replace the sample CSV with your own trade history.

Example:

```
tests/
    fixtures/
        my_strategy.csv
```

Run:

```bash
python -m gate validate \
    --trades tests/fixtures/my_strategy.csv
```

You may also keep your strategy anywhere on disk:

```bash
python -m gate validate \
    --trades E:\Strategies\eurusd.csv
```

---

# Changing Validation Parameters

The framework allows every statistical component to be configured.

Example

```bash
python -m gate validate \
    --trades strategy.csv \
    --window-size 100 \
    --step-size 25 \
    --configs-tried 50 \
    --null-iterations 10000
```

### Available Parameters

| Parameter | Description | Default |
|-----------|-------------|----------|
| `--window-size` | Walk-forward window size | 50 |
| `--step-size` | Sliding window step | 50 |
| `--configs-tried` | Number of tested strategies | Optional |
| `--null-iterations` | Permutation iterations | 5000 |
| `--bootstrap-iterations` | Bootstrap samples | 5000 |

---

# Validation Output

Running the framework produces a complete statistical report.

```
SIGNAL VALIDATION GATE

Walk Forward Distribution

Observed Trade Metrics

Bootstrap Lower Bounds

Selection Adjusted Null

Verdict

Message
```

Possible verdicts

```
EDGE

NO-EDGE

INSUFFICIENT DATA
```

---

# Code Quality Checks

Run Ruff

```bash
ruff check gate
```

Automatically fix formatting

```bash
ruff check gate --fix
```

Run static type checking

```bash
mypy gate
```

---

# Running Individual Components

Bootstrap

```python
from gate.bootstrap import run_bootstrap
```

Walk Forward

```python
from gate.windows import calculate_walk_forward_summary
```

Observed Metrics

```python
from gate.metrics import calculate_trade_metrics
```

Selection-Adjusted Null

```python
from gate.nulls import run_selection_adjusted_null
```

Final Verdict

```python
from gate.verdict import evaluate_verdict
```

Validation Pipeline

```python
from gate import validate
```

---

# Typical Development Workflow

```text
Modify CSV
      ↓
Run Validation
      ↓
Inspect Metrics
      ↓
Inspect Bootstrap
      ↓
Inspect Null Test
      ↓
Review Final Verdict
      ↓
Refine Strategy
      ↓
Repeat
```

---

# Troubleshooting

## Virtual environment not activating

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

---

## Ruff reports import errors

```bash
ruff check gate --fix
```

---

## Mypy reports type errors

```bash
mypy gate
```

---

## CSV validation fails

Verify that your CSV includes:

- `open_time`
- `close_time`
- `side`
- `return_R`

and contains completed trades.

---

# Next Steps

Once your strategy consistently receives an **EDGE** verdict, it is ready to be used as high-quality training data for machine learning models.
---

# 16. Future Improvements

Planned enhancements include:

- Unit testing with pytest
- HTML report generation
- PDF export
- JSON output mode
- YAML configuration
- Interactive dashboard
- Batch strategy validation
- Parallel bootstrap execution
- REST API
- Web interface
- CI/CD integration
- Performance benchmarking

---

# Conclusion

The **Signal Validation Gate** provides a complete, modular, and statistically rigorous framework for evaluating quantitative trading strategies before machine learning training.

By combining:

- Input validation
- Strategy normalization
- Walk-forward analysis
- Observed trade metrics
- Bootstrap confidence estimation
- Selection-adjusted permutation testing
- Rule-based verdict evaluation

the framework ensures that only statistically supported trading strategies progress to downstream modeling.

The modular architecture, strong separation of concerns, immutable data models, and layered pipeline design make the system easy to extend, test, and maintain while adhering to modern software engineering best practices.