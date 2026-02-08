# EuraGovExam

**A Multilingual Multimodal Benchmark from Real-World Civil Service Exams**

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org)
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
# Run evaluation on full benchmark (default: image-only)
python experiment/evaluate.py --model gemini-2.0-flash

# Evaluate on specific nation
python experiment/evaluate.py --nation japan

# Evaluate on specific domain
python experiment/evaluate.py --domain mathematics

# Combine filters
python experiment/evaluate.py --nation korea --domain law
```

### Evaluation Protocol

**Image-Only Evaluation** (default)
- Model receives only the exam image
- No external OCR or text extraction
- Minimal standardized instruction
- Measures combined visual perception + reasoning

```bash
python experiment/evaluate.py --model gemini-2.0-flash
```

For advanced multi-track analysis (text-only, multimodal, VCE), see [`experiment/EVALUATE_README.md`](experiment/EVALUATE_README.md).

## Random Baseline

The benchmark's random baseline is **23.7%** (weighted average of 4-choice and 5-choice questions).

## Files

- **`experiment/evaluate.py`**: Main evaluation script
- **`experiment/EVALUATE_README.md`**: Detailed usage guide
- **`experiment/config.py`**: Configuration (API keys, model settings)
- **`experiment/requirements.txt`**: Python dependencies
- **`experiment/analysis/`**: Statistical analysis scripts
- **`experiment/figures/`**: Generated figures

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
