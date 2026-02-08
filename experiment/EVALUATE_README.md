# EuraGovExam Evaluation Script

## Overview

`evaluate.py` provides standardized evaluation for the EuraGovExam benchmark using the Image-Only protocol.

## Quick Start

```bash
# Full benchmark evaluation
python evaluate.py --model gemini-2.0-flash

# Filter by nation
python evaluate.py --nation japan

# Filter by domain
python evaluate.py --domain mathematics

# Combine filters
python evaluate.py --nation korea --domain law
```

## Evaluation Protocol

**Image-Only Evaluation**
- Model receives only the exam image
- No external OCR or text extraction
- Minimal standardized instruction
- Measures combined visual perception + reasoning

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model identifier | `gemini-2.0-flash` |
| `--split` | Dataset split (`train`, `test`) | `train` |
| `--nation` | Filter by nation (`korea`, `japan`, `taiwan`, `india`, `eu`) | None (all) |
| `--domain` | Filter by domain/subject | None (all) |
| `--sample-size` | Override sample count | None (use all) |
| `--output-dir` | Result JSON directory | `results/` |
| `--seed` | Random seed for reproducibility | `42` |
| `--verbose` | Enable verbose logging | False |

### Valid Domains

`mathematics`, `physics`, `chemistry`, `biology`, `earth_science`, `history`, `geography`, `politics`, `economics`, `law`, `sociology`, `ethics`, `language`, `literature`, `art`, `music`, `physical_education`

## Output Format

Results are saved as JSON:

```json
{
  "metadata": {
    "model": "gemini-2.0-flash",
    "split": "train",
    "filters": {"nation": "japan", "domain": null},
    "sample_size": 100,
    "seed": 42
  },
  "metrics": {
    "overall_accuracy": 67.5,
    "random_baseline": 23.7,
    "by_nation": {...},
    "by_domain": {...}
  },
  "results": [...]
}
```

## Examples

```bash
# Basic usage
python evaluate.py

# Quick test
python evaluate.py --sample-size 10 --verbose

# Filter by nation
python evaluate.py --nation japan

# Filter by domain
python evaluate.py --domain mathematics

# Combined filters
python evaluate.py --nation korea --domain law --sample-size 50
```

## Reproducibility

```bash
# Run 1
python evaluate.py --sample-size 100 --seed 42 --output-dir run1

# Run 2 (identical results)
python evaluate.py --sample-size 100 --seed 42 --output-dir run2

# Verify
diff <(jq '.results' run1/*.json) <(jq '.results' run2/*.json)
```

## Random Baseline

The benchmark's random baseline is **23.7%** (weighted average of 4-choice and 5-choice questions).
