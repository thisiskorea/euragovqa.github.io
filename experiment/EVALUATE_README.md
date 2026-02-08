# EuraGovExam Standardized Evaluation Script

## Overview

`evaluate.py` provides a standardized, reproducible evaluation interface for the EuraGovExam benchmark, following the paper's evaluation protocol.

## Quick Start

```bash
# Full benchmark evaluation (Image-Only Setting)
python evaluate.py --model gemini-2.0-flash --setting image-only

# Filter by nation
python evaluate.py --model gemini-2.0-flash --nation japan

# Filter by domain
python evaluate.py --model gemini-2.0-flash --domain mathematics

# Combine filters
python evaluate.py --model gemini-2.0-flash --nation korea --domain law
```

## Evaluation Settings

The script supports three evaluation tracks:

### 1. Image-Only (Default, Track A)
- **Primary evaluation mode** as described in the paper
- Model receives only: standardized instruction + exam image
- No external OCR allowed
- Measures combined perception + reasoning capability

```bash
python evaluate.py --setting image-only
```

### 2. Text-Only (Track B)
- OCR extraction followed by text-only reasoning
- Measures pure reasoning on noisy OCR text
- Useful for isolating perception vs reasoning bottlenecks

```bash
python evaluate.py --setting text-only
```

### 3. Multimodal (Track C)
- Model receives both image and OCR text
- Measures multimodal fusion capability
- Compares against Track A and B for VCE analysis

```bash
python evaluate.py --setting multimodal
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Model identifier | `gemini-2.0-flash` |
| `--split` | Dataset split (`train`, `test`) | `train` |
| `--setting` | Evaluation setting (`image-only`, `text-only`, `multimodal`) | `image-only` |
| `--nation` | Filter by nation (`korea`, `japan`, `taiwan`, `india`, `eu`) | None (all) |
| `--domain` | Filter by domain/subject (see below) | None (all) |
| `--sample-size` | Override sample count | None (use all) |
| `--output-dir` | Result JSON directory | `results/` |
| `--seed` | Random seed for reproducibility | `42` |
| `--verbose` | Enable verbose logging | False |

### Valid Domains

- `mathematics`
- `physics`
- `chemistry`
- `biology`
- `earth_science`
- `history`
- `geography`
- `politics`
- `economics`
- `law`
- `sociology`
- `ethics`
- `language`
- `literature`
- `art`
- `music`
- `physical_education`

## Output Format

Results are saved as JSON files with the following structure:

```json
{
  "metadata": {
    "model": "gemini-2.0-flash",
    "setting": "image-only",
    "split": "train",
    "filters": {
      "nation": "japan",
      "domain": null
    },
    "sample_size": 100,
    "seed": 42,
    "timestamp": "20260209_143022"
  },
  "metrics": {
    "overall_accuracy": 67.5,
    "random_baseline": 23.7,
    "correct_count": 67,
    "total_count": 100,
    "by_nation": {
      "Japan": {
        "accuracy": 67.5,
        "correct": 67,
        "total": 100
      }
    },
    "by_domain": {
      "mathematics": {
        "accuracy": 75.0,
        "correct": 15,
        "total": 20
      },
      ...
    }
  },
  "results": [
    {
      "index": 123,
      "nation": "Japan",
      "task": "physics",
      "correct_answer": "C",
      "predicted_answer": "C",
      "is_correct": true,
      "response": "..."
    },
    ...
  ]
}
```

## Evaluation Protocol

### Standardized Instruction

The script uses the minimal instruction specified in the paper (Track A):

```
You are solving a multiple-choice exam question shown in the image.
Carefully read the question and all answer options.
At the very end, provide the final answer in exactly this format:
The answer is X. (For example: The answer is B.)
```

### Answer Extraction

Strict 4-level regex cascade:
1. `"The answer is X."`
2. `"Answer: X"`
3. Standalone letter at end of line
4. Last capital letter (fallback)

Format violations → marked as **INVALID** (incorrect)

### Random Baseline

The benchmark's random baseline is **23.7%** (weighted average of 4-choice and 5-choice questions).

## Examples

### Basic Usage

```bash
# Evaluate with default settings (full dataset, image-only)
python evaluate.py

# Small sample for quick testing
python evaluate.py --sample-size 10 --verbose
```

### Filtering by Region

```bash
# Evaluate on Japanese exam questions
python evaluate.py --nation japan

# Evaluate on South Korean questions
python evaluate.py --nation korea
```

### Filtering by Domain

```bash
# Evaluate on mathematics questions
python evaluate.py --domain mathematics

# Evaluate on law questions
python evaluate.py --domain law
```

### Combined Filtering

```bash
# Korean law questions
python evaluate.py --nation korea --domain law

# Japanese physics questions
python evaluate.py --nation japan --domain physics --sample-size 50
```

### Multi-Track Comparison

```bash
# Run all three tracks on the same sample
python evaluate.py --setting image-only --sample-size 100 --seed 42
python evaluate.py --setting text-only --sample-size 100 --seed 42
python evaluate.py --setting multimodal --sample-size 100 --seed 42
```

## Reproducibility

To ensure reproducible results:

1. Use the same `--seed` value
2. Use the same `--sample-size` (or omit for full dataset)
3. Use the same filters (`--nation`, `--domain`)

```bash
# Run 1
python evaluate.py --sample-size 100 --seed 42 --output-dir run1

# Run 2 (should produce identical results)
python evaluate.py --sample-size 100 --seed 42 --output-dir run2

# Verify reproducibility
diff <(jq '.results' run1/*.json) <(jq '.results' run2/*.json)
```

## Analyzing Results

```bash
# View overall accuracy
jq '.metrics.overall_accuracy' results/evaluate_*.json

# View by-nation breakdown
jq '.metrics.by_nation' results/evaluate_*.json

# View by-domain breakdown
jq '.metrics.by_domain' results/evaluate_*.json

# Compare with random baseline
jq 'if .metrics.overall_accuracy < 23.7
    then "BELOW BASELINE"
    else "ABOVE BASELINE"
    end' results/evaluate_*.json

# Count samples
jq '.results | length' results/evaluate_*.json

# Verify filtering
jq '.results[].nation | unique' results/evaluate_*_japan_*.json
```

## Performance Notes

- **API Rate Limiting**: 2-second delay between API calls
- **Retry Logic**: Exponential backoff with 2 max retries
- **Progress Tracking**: Prints every 10 samples
- **Large Evaluations**: Full dataset (8000 items) takes ~5-6 hours

For large-scale experiments, consider:
- Using `--sample-size` for pilot studies
- Running overnight for full evaluations
- Using filters to evaluate specific subsets

## Troubleshooting

### No samples match filters

```
WARNING: No samples match filters (nation=korea, domain=physics)
```

Check that the domain name is correct (use `--help` to see valid domains) and that the nation-domain combination exists in the dataset.

### API errors

If you encounter rate limiting or API errors, the script will:
- Automatically retry with exponential backoff
- Log errors to console
- Mark failed samples with "ERROR: ..." in response field

### Memory issues

For very large evaluations, monitor memory usage. The script loads the full dataset into memory but processes samples sequentially.

## Citation

If you use this evaluation script, please cite the EuraGovExam paper:

```bibtex
@inproceedings{euragovexam2025,
  title={EuraGovExam: A Multilingual Multimodal Benchmark for Vision-Language Models},
  author={...},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2025}
}
```
