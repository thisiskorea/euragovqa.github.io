# EuraGovExam Evaluation

Two self-contained scripts for evaluating VLMs on the EuraGovExam benchmark.

| Script | Purpose |
|--------|---------|
| `run_api.py` | Cloud API evaluation (Gemini, OpenAI) |
| `run_hf.py` | Local HuggingFace model evaluation |

## Installation

```bash
pip install -r experiment/requirements.txt
```

> **Note**: `run_hf.py` additionally requires `torch`, `transformers`, and `accelerate`. If you only need API evaluation, you can skip those.

## run_api.py — Cloud API Evaluation

### Quick Start

```bash
# Gemini
python experiment/run_api.py --provider gemini --model gemini-2.0-flash

# OpenAI
python experiment/run_api.py --provider openai --model gpt-4o
```

### API Key

Provide via `--api-key` flag or environment variable:

```bash
# Environment variable
export GEMINI_API_KEY="your-key"
python experiment/run_api.py --provider gemini --model gemini-2.0-flash

# Command-line flag
python experiment/run_api.py --provider openai --model gpt-4o --api-key sk-...
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | `gemini` or `openai` | (required) |
| `--model` | Model identifier | (required) |
| `--api-key` | API key | env var |
| `--split` | `train` or `test` | `train` |
| `--nation` | `korea`, `japan`, `taiwan`, `india`, `eu` | all |
| `--domain` | Filter by domain/task | all |
| `--sample-size` | Override sample count | all |
| `--output-dir` | Result directory | `results` |
| `--seed` | Random seed | `42` |
| `--verbose` | Verbose logging | off |
| `--delay` | Seconds between API calls | `2.0` |
| `--max-retries` | Max retry attempts | `2` |

### Examples

```bash
# Filter by nation
python experiment/run_api.py --provider gemini --model gemini-2.0-flash --nation japan

# Filter by domain
python experiment/run_api.py --provider gemini --model gemini-2.0-flash --domain mathematics

# Combined filters with sample size
python experiment/run_api.py --provider gemini --model gemini-2.0-flash --nation korea --domain law --sample-size 50

# Quick verbose test
python experiment/run_api.py --provider gemini --model gemini-2.0-flash --sample-size 10 --verbose
```

## run_hf.py — Local HuggingFace Model Evaluation

### Quick Start

```bash
python experiment/run_hf.py --model Qwen/Qwen2-VL-7B-Instruct
python experiment/run_hf.py --model llava-hf/llava-1.5-7b-hf --dtype float16
```

### Supported Models

Explicitly supported architectures (auto-detected from model ID):

| Architecture | Example Models |
|-------------|----------------|
| Qwen2-VL / Qwen2.5-VL | `Qwen/Qwen2-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct` |
| LLaVA 1.5 | `llava-hf/llava-1.5-7b-hf`, `llava-hf/llava-1.5-13b-hf` |
| LLaVA-NeXT | `llava-hf/llava-v1.6-mistral-7b-hf` |

Other VLMs (InternVL, Phi-3-vision, Llama-Vision, etc.) are loaded via `AutoModelForVision2Seq` with `trust_remote_code=True`.

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | HuggingFace model ID | (required) |
| `--split` | `train` or `test` | `train` |
| `--nation` | `korea`, `japan`, `taiwan`, `india`, `eu` | all |
| `--domain` | Filter by domain/task | all |
| `--sample-size` | Override sample count | all |
| `--output-dir` | Result directory | `results` |
| `--seed` | Random seed | `42` |
| `--verbose` | Verbose logging | off |
| `--device` | `auto`, `cuda`, `cuda:0`, `cpu` | `auto` |
| `--dtype` | `auto`, `float16`, `bfloat16`, `float32` | `auto` |
| `--max-new-tokens` | Max generation length | `512` |

### Examples

```bash
# Basic usage
python experiment/run_hf.py --model Qwen/Qwen2-VL-7B-Instruct

# Specify dtype and device
python experiment/run_hf.py --model llava-hf/llava-1.5-7b-hf --dtype float16 --device cuda:0

# Filter and sample
python experiment/run_hf.py --model Qwen/Qwen2-VL-7B-Instruct --nation japan --sample-size 100

# Verbose mode
python experiment/run_hf.py --model Qwen/Qwen2-VL-7B-Instruct --sample-size 10 --verbose
```

## Output Format

Both scripts produce identical JSON output:

```json
{
  "metadata": {
    "model": "gemini-2.0-flash",
    "provider": "gemini",
    "split": "train",
    "filters": {"nation": null, "domain": null},
    "sample_size": 8000,
    "seed": 42,
    "timestamp": "20260225_143022"
  },
  "metrics": {
    "overall_accuracy": 67.5,
    "random_baseline": 23.7,
    "correct_count": 5400,
    "total_count": 8000,
    "by_nation": {"South Korea": {"accuracy": 65.2, "correct": 1596, "total": 2448}},
    "by_domain": {"mathematics": {"accuracy": 70.1, "correct": 350, "total": 499}}
  },
  "results": [
    {
      "index": 0,
      "nation": "South Korea",
      "task": "mathematics",
      "correct_answer": "C",
      "predicted_answer": "C",
      "is_correct": true,
      "response": "..."
    }
  ]
}
```

## Reproducibility

```bash
# Run 1
python experiment/run_api.py --provider gemini --model gemini-2.0-flash --sample-size 100 --seed 42 --output-dir run1

# Run 2 (identical sampling)
python experiment/run_api.py --provider gemini --model gemini-2.0-flash --sample-size 100 --seed 42 --output-dir run2
```

## Random Baseline

The benchmark's random baseline is **23.7%** (weighted average of 4-choice and 5-choice questions).
