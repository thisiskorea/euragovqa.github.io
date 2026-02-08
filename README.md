# EuraGovExam

**A Multilingual Multimodal Benchmark for Vision-Language Models**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/EuraGovExam/EuraGovExam)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue)](LICENSE)

## Overview

EuraGovExam is a benchmark that evaluates Vision-Language Models (VLMs) on authentic civil service exam questions from 5 Eurasian regions: **South Korea, Japan, Taiwan, India, and the European Union**. The benchmark is designed to diagnose perception vs. reasoning bottlenecks in VLMs through a rigorous three-track evaluation protocol.

### Key Features

- **8,000 authentic exam questions** from government exams
- **5 regions** with diverse languages and scripts (Korean, Japanese, Chinese, English, European languages)
- **17 domains** covering STEM, humanities, and social sciences
- **Three-track evaluation** to isolate visual perception from reasoning
- **Standardized evaluation protocol** for reproducible results

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
# Run standardized evaluation on full benchmark
python experiment/evaluate.py --model gemini-2.0-flash --setting image-only

# Evaluate on specific nation
python experiment/evaluate.py --model gemini-2.0-flash --nation japan

# Evaluate on specific domain
python experiment/evaluate.py --model gemini-2.0-flash --domain mathematics

# Combine filters
python experiment/evaluate.py --model gemini-2.0-flash --nation korea --domain law
```

For detailed usage, see [`experiment/EVALUATE_README.md`](experiment/EVALUATE_README.md).

## Three-Track Evaluation Protocol

EuraGovExam uses three evaluation tracks to isolate VLM capabilities:

### Track A: Image-Only (Primary)
- **What it measures**: Combined visual perception + reasoning
- **Input**: Exam image only (no external OCR)
- **Use case**: Standard benchmark evaluation

```bash
python experiment/evaluate.py --setting image-only
```

### Track B: Text-Only
- **What it measures**: Pure language reasoning on noisy OCR text
- **Input**: Extracted OCR text only (no image)
- **Use case**: Isolate reasoning capability

```bash
python experiment/evaluate.py --setting text-only
```

### Track C: Multimodal
- **What it measures**: Multimodal fusion capability
- **Input**: Both image and OCR text
- **Use case**: Measure fusion benefit/interference

```bash
python experiment/evaluate.py --setting multimodal
```

### Visual Causal Effect (VCE)

**VCE = Acc(Track C) - Acc(Track B)**

- **Positive VCE**: Visual input helps (fusion benefit)
- **Negative VCE**: Visual input hurts (fusion interference)
- **Zero VCE**: No fusion effect

## Evaluation Results

### Random Baseline
The benchmark's random baseline is **23.7%** (weighted average of 4-choice and 5-choice questions).

### Sample Results

| Model | Overall | Korea | Japan | Taiwan | India | EU |
|-------|---------|-------|-------|--------|-------|-----|
| GPT-4o | 67.5% | 65.2% | 68.9% | 71.2% | 64.8% | 69.3% |
| Claude-3.5 | 65.3% | 63.1% | 66.8% | 69.5% | 62.7% | 67.2% |
| Gemini 2.0 Flash | 63.8% | 61.5% | 65.2% | 68.1% | 60.9% | 65.7% |

*Note: These are example results. Run experiments to get actual numbers.*

## Experiment Scripts

The `experiment/` directory contains various scripts for running experiments:

### Core Scripts
- **`evaluate.py`**: Standardized evaluation script (recommended)
- **`config.py`**: Configuration (API keys, model settings, prompts)
- **`requirements.txt`**: Python dependencies

### Experiment Scripts
- `quick_test.py`: Quick validation with 5 samples
- `large_scale_experiment.py`: Large-scale 3-track experiment
- `multi_model_experiment.py`: Multi-model comparison
- `vce_analysis.py`: Visual Causal Effect analysis

### Analysis Scripts (`experiment/analysis/`)
- `phase1_statistical_analysis.py`: Bootstrap CI, significance tests
- `phase2_failure_taxonomy.py`: Qualitative failure categorization
- `phase3_clean_text_experiment.py`: Oracle text experiments
- `mixed_effects_anova.py`: Track × Region interaction analysis

### Figure Generation
- `generate_figures.py`: Generate publication figures
- `paper_figures_tables.py`: Generate tables for paper

## Configuration

Edit `experiment/config.py` to configure:

```python
# API Keys
GEMINI_API_KEY = "your-api-key-here"

# Model settings
MODEL_NAME = "gemini-2.0-flash"

# Dataset settings
DATASET_NAME = "EuraGovExam/EuraGovExam"
DATASET_SPLIT = "train"

# Prompts (standardized)
PROMPT_TRACK_A = """You are solving a multiple-choice exam question..."""
```

## Output Format

Results are saved as JSON:

```json
{
  "metadata": {
    "model": "gemini-2.0-flash",
    "setting": "image-only",
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
python experiment/evaluate.py --sample-size 100 --seed 42 --output-dir run1

# Run 2 (identical results)
python experiment/evaluate.py --sample-size 100 --seed 42 --output-dir run2

# Verify
diff <(jq '.results' run1/*.json) <(jq '.results' run2/*.json)
```

## Paper and Citation

If you use EuraGovExam in your research, please cite:

```bibtex
@inproceedings{euragovexam2025,
  title={EuraGovExam: A Multilingual Multimodal Benchmark for Vision-Language Models},
  author={[Authors]},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2025}
}
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
│   ├── evaluate.py                    # Main evaluation script
│   ├── EVALUATE_README.md             # Detailed usage guide
│   ├── config.py                      # Configuration
│   ├── requirements.txt               # Dependencies
│   ├── analysis/                      # Analysis scripts
│   │   ├── phase1_statistical_analysis.py
│   │   ├── phase2_failure_taxonomy.py
│   │   └── ...
│   ├── figures/                       # Generated figures
│   └── paper_draft/                   # Paper manuscript
├── data/                              # Dataset samples
├── assets/                            # Static assets
└── static/                            # Web assets
```

## Support

For questions, issues, or contributions:
- **Issues**: https://github.com/thisiskorea/EuraGovExam/issues
- **Dataset**: https://huggingface.co/datasets/EuraGovExam/EuraGovExam

## Acknowledgments

We thank the governments of South Korea, Japan, Taiwan, India, and the European Union for making their civil service exam materials publicly available for educational and research purposes.

---

**Website**: [Coming Soon]
**Leaderboard**: [Coming Soon]
**Paper**: [Coming Soon]