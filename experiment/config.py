# EuraGovExam 3-Track 실험 설정

# API Keys
GEMINI_API_KEY = ""

# Model settings
MODEL_NAME = "gemini-2.0-flash"  # Gemini 2.0 Flash

# Dataset settings
DATASET_NAME = "EuraGovExam/EuraGovExam"
DATASET_SPLIT = "train"

# Experiment settings
PILOT_SAMPLE_SIZE = 100
MAIN_SAMPLE_SIZE = 1000

# Sampling distribution (matches dataset distribution)
NATION_DISTRIBUTION = {
    "South Korea": 0.306,
    "Japan": 0.256,
    "EU": 0.240,
    "India": 0.128,
    "Taiwan": 0.070,
}

# Prompts
PROMPT_TRACK_A = """You are solving a multiple-choice exam question shown in the image.
Carefully read the question and all answer options.
At the very end, provide the final answer in exactly this format:
The answer is X. (For example: The answer is B.)"""

PROMPT_TRACK_B = """You are solving a multiple-choice exam question.
The question text extracted from a scanned exam document is provided below.

---
{ocr_text}
---

Carefully read the question and all answer options.
At the very end, provide the final answer in exactly this format:
The answer is X. (For example: The answer is B.)"""

PROMPT_TRACK_C = """You are solving a multiple-choice exam question shown in the image.
For your reference, here is the text extracted from the image:

---
{ocr_text}
---

Use both the image and the extracted text to answer the question.
At the very end, provide the final answer in exactly this format:
The answer is X. (For example: The answer is B.)"""

# Output settings
OUTPUT_DIR = "results"
