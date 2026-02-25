# EuraGovExam

**A Multilingual Multimodal Benchmark from Real-World Civil Service Exams**

[![Website](https://img.shields.io/badge/Website-red)](https://thisiskorea.github.io/EuraGovExam/index.html)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/EuraGovExam/EuraGovExam)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue)](LICENSE)

## Overview

EuraGovExam is a benchmark that evaluates Vision-Language Models (VLMs) on authentic civil service exam questions from 5 Eurasian regions: **South Korea, Japan, Taiwan, India, and the European Union**. The benchmark is designed to diagnose perception vs. reasoning bottlenecks in VLMs through a rigorous three-track evaluation protocol.

### Key Features

- **8,000 authentic exam questions** from government civil service exams
- **5 regions** with diverse languages and scripts (Korean, Japanese, Chinese, English, European languages)
- **17 domains** covering STEM, humanities, and social sciences
- **Image-only evaluation** with minimal instruction (no external OCR)
- **Standardized protocol** for reproducible results

## Dataset

The dataset is hosted on HuggingFace: [`EuraGovExam/EuraGovExam`](https://huggingface.co/datasets/EuraGovExam/EuraGovExam)

### Statistics

| Region | Questions | Percentage |
|--------|-----------|------------|
| South Korea | 2,448 | 30.6% |
| Japan | 2,048 | 25.6% |
| EU | 1,920 | 24.0% |
| India | 1,024 | 12.8% |
| Taiwan | 560 | 7.0% |
| **Total** | **8,000** | **100%** |

### Domains

Mathematics, Physics, Chemistry, Biology, Earth Science, History, Geography, Politics, Economics, Law, Sociology, Ethics, Language, Literature, Art, Music, Physical Education

## Quick Start

### Installation

```bash
git clone https://github.com/thisiskorea/EuraGovExam.git
cd EuraGovExam
pip install -r experiment/requirements.txt
```

### Basic Usage

```bash
# Cloud API evaluation (Gemini)
python experiment/run_api.py --provider gemini --model gemini-2.0-flash

# Cloud API evaluation (OpenAI)
python experiment/run_api.py --provider openai --model gpt-4o

# Local HuggingFace model evaluation
python experiment/run_hf.py --model Qwen/Qwen2-VL-7B-Instruct

# Filter by nation or domain
python experiment/run_api.py --provider gemini --model gemini-2.0-flash --nation japan
python experiment/run_hf.py --model Qwen/Qwen2-VL-7B-Instruct --domain mathematics
```

### Evaluation Protocol

**Image-Only Evaluation** (default)
- Model receives only the exam image
- No external OCR or text extraction
- Minimal standardized instruction
- Measures combined visual perception + reasoning

For full CLI options and supported models, see [`experiment/EVALUATE_README.md`](experiment/EVALUATE_README.md).

## Random Baseline

The benchmark's random baseline is **23.7%** (weighted average of 4-choice and 5-choice questions).

## Files

- **`experiment/run_api.py`**: Cloud API evaluation (Gemini, OpenAI)
- **`experiment/run_hf.py`**: Local HuggingFace model evaluation
- **`experiment/EVALUATE_README.md`**: Detailed usage guide
- **`experiment/requirements.txt`**: Python dependencies

## Configuration

API keys can be provided via command-line arguments or environment variables:

```bash
# Via environment variable
export GEMINI_API_KEY="your-key-here"
python experiment/run_api.py --provider gemini --model gemini-2.0-flash

# Via command-line argument
python experiment/run_api.py --provider openai --model gpt-4o --api-key sk-...
```

## Output Format

Results are saved as JSON:

```json
{
  "metadata": {
    "model": "gemini-2.0-flash",
    "provider": "gemini",
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
  "results": [
    {
      "index": 123,
      "nation": "Japan",
      "task": "physics",
      "correct_answer": "C",
      "predicted_answer": "C",
      "is_correct": true
    }
  ]
}
```

## Reproducibility

All experiments are reproducible with random seeds:

```bash
# Run 1
python experiment/run_api.py --provider gemini --model gemini-2.0-flash --sample-size 100 --seed 42 --output-dir run1

# Run 2 (identical sampling)
python experiment/run_api.py --provider gemini --model gemini-2.0-flash --sample-size 100 --seed 42 --output-dir run2
```



## License

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

- **Attribution**: Credit must be given to the creators
- **NonCommercial**: Only noncommercial uses allowed
- **ShareAlike**: Adaptations must be shared under the same license

## Project Structure

```
EuraGovExam/
├── README.md                          # This file
├── experiment/
│   ├── run_api.py                     # Cloud API evaluation (Gemini, OpenAI)
│   ├── run_hf.py                      # Local HuggingFace model evaluation
│   ├── EVALUATE_README.md             # Detailed usage guide
│   └── requirements.txt               # Dependencies
├── data/                              # Dataset samples
├── assets/                            # Static assets
└── static/                            # Web assets
```


