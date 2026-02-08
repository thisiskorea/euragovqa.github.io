# Section 4: Evaluation Protocol (한국어 초안)

---

## 4.1 Task Definition

### 평가 과제 정의

EuraGovExam의 평가 과제는 **Image-Only Multiple Choice Question Answering**이다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    TASK DEFINITION                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input:  Exam question image (scanned document)                 │
│  Output: Answer choice (1, 2, 3, 4, or 5)                       │
│                                                                 │
│  Constraints:                                                   │
│  • No external text prompts (question is in image only)         │
│  • No OCR tool usage allowed                                    │
│  • No external knowledge retrieval                              │
│  • Single image per question                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 프롬프트 설계

모든 모델에 동일한 프롬프트를 사용:

```
[System Prompt]
You are taking a civil service examination. Look at the exam question 
image carefully and select the correct answer from the given choices.
Respond with only the answer number (1, 2, 3, 4, or 5).

[User Prompt]
<image>
What is the correct answer?
```

### 평가 기준

- **정확도(Accuracy)**: 정답 일치 여부
- **정답 파싱**: 모델 출력에서 숫자 추출 (regex: `[1-5]`)
- **무응답 처리**: 파싱 실패 시 오답 처리

---

## 4.2 Evaluated Models

### 모델 선정 기준

1. **최신성**: 2024-2025년 공개된 최신 모델 우선
2. **다양성**: Closed/Open, 다양한 규모, 다양한 아키텍처
3. **재현성**: API 또는 공개 가중치로 접근 가능

### 평가 모델 목록 (23개)

**Closed-source Models (9개)**

| Model | Provider | Release | Notes |
|-------|----------|---------|-------|
| Gemini-2.5-pro | Google | 2025 | Best overall |
| Gemini-2.5-flash | Google | 2025 | Fast, efficient |
| Gemini-2.5-flash-lite | Google | 2025 | Lightweight |
| o3 | OpenAI | 2025 | Reasoning-focused |
| o4-mini | OpenAI | 2025 | Reasoning-focused |
| GPT-4o | OpenAI | 2024 | Multimodal flagship |
| GPT-4.1 | OpenAI | 2025 | Updated GPT-4 |
| GPT-4.1-mini | OpenAI | 2025 | Efficient variant |
| Claude-Sonnet-4 | Anthropic | 2025 | Latest Claude |

**Open-source Models (14개)**

| Model | Parameters | Release | Notes |
|-------|------------|---------|-------|
| Qwen2-VL-72B-Instruct | 72B | 2024 | Best open model |
| Qwen2-VL-7B-Instruct | 7B | 2024 | Mid-size Qwen |
| Qwen2-VL-2B-Instruct | 2B | 2024 | Small Qwen |
| Qwen2.5-VL-7B-Instruct | 7B | 2025 | Updated Qwen |
| InternVL2.5-38B-MPO | 38B | 2024 | Large open VLM |
| Ovis2-32B | 32B | 2024 | |
| Ovis2-16B | 16B | 2024 | |
| Ovis2-8B | 8B | 2024 | |
| LLaVA-1.5-13B | 13B | 2023 | Classic baseline |
| LLaVA-1.5-7B | 7B | 2023 | Small baseline |
| Llama-3.2-11B-Vision | 11B | 2024 | Meta's VLM |
| llama3-llava-next-8b | 8B | 2024 | |
| LLaVA-NeXT-Video-7B-DPO | 7B | 2024 | |
| Phi-3.5-vision-instruct | 4B | 2024 | Microsoft small |

---

## 4.3 Inference Settings

### 추론 설정

**공통 설정:**
```yaml
temperature: 0.0  # Deterministic
max_tokens: 10    # Only need answer number
top_p: 1.0
```

**이미지 처리:**
- 원본 해상도 유지 (리사이징 없음)
- PNG 포맷 (손실 없는 압축)
- API별 이미지 인코딩 방식 준수

**API 호출:**
- Closed models: 공식 API 사용
- Open models: vLLM 또는 Transformers 사용
- GPU: NVIDIA A100 80GB

---

## 4.4 Diagnostic 4-Track Protocol

### 진단 프로토콜 개요

VLM 실패의 원인을 분석하기 위해, 4가지 Track으로 평가를 수행한다:

```
┌─────────────────────────────────────────────────────────────────┐
│                 4-TRACK DIAGNOSTIC PROTOCOL                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                      ┌──────────────┐                           │
│                      │  Exam Image  │                           │
│                      └──────┬───────┘                           │
│                             │                                   │
│         ┌───────────────────┼───────────────────┐               │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│   ┌──────────┐       ┌──────────┐       ┌──────────┐           │
│   │ TRACK A  │       │ TRACK B  │       │ TRACK C  │           │
│   │ Image    │       │ Text     │       │ Image +  │           │
│   │ Only     │       │ Only     │       │ Text     │           │
│   │   ↓      │       │   ↓      │       │   ↓      │           │
│   │  VLM     │       │  LLM     │       │  VLM     │           │
│   └────┬─────┘       └────┬─────┘       └────┬─────┘           │
│        │                  │                  │                  │
│        │            ┌─────┴─────┐            │                  │
│        │            │           │            │                  │
│        │      ┌─────┴─────┐ ┌───┴───┐       │                  │
│        │      │ Track B1  │ │Track B2│       │                  │
│        │      │ OCR Text  │ │Oracle  │       │                  │
│        │      │    ↓      │ │ Text   │       │                  │
│        │      │   LLM     │ │   ↓    │       │                  │
│        │      └─────┬─────┘ │  LLM   │       │                  │
│        │            │       └───┬───┘       │                  │
│        ▼            ▼           ▼            ▼                  │
│   ┌─────────────────────────────────────────────────┐          │
│   │                 DELTA ANALYSIS                   │          │
│   └─────────────────────────────────────────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Track 상세 설명

**Track A: Image Only (Main Evaluation)**
- 입력: 시험 이미지만
- 모델: VLM
- 측정: VLM의 종합적 능력 (OCR + 이해 + 추론)

**Track B1: OCR → LLM**
- 입력: OCR로 추출한 텍스트
- 모델: LLM (텍스트 전용)
- 측정: 읽기 병목 제거 시 성능
- OCR 엔진: Gemini-2.0-Flash (내장 OCR)

**Track B2: Oracle Text → LLM**
- 입력: 인간이 정확히 타이핑한 텍스트 (subset)
- 모델: LLM
- 측정: 완벽한 읽기 가정 시 성능 상한선

**Track C: Image + Text**
- 입력: 시험 이미지 + OCR 텍스트
- 모델: VLM
- 측정: 멀티모달 시너지 효과

### Delta 분석

Track 간 성능 차이로 병목 원인을 분석:

| Delta | Calculation | Interpretation |
|-------|-------------|----------------|
| Δ(B1-A) | Track B1 - Track A | > 0: VLM 내장 OCR이 병목 |
| Δ(B2-B1) | Track B2 - Track B1 | > 0: OCR 엔진 품질이 문제 |
| Δ(B2-A) | Track B2 - Track A | > 0: 전체 perception 병목 |
| Δ(C-A) | Track C - Track A | > 0: 멀티모달 시너지 존재 |
| Δ(C-B2) | Track C - Track B2 | > 0: 이미지가 추가 가치 제공 |

---

## 4.5 Metrics and Statistical Analysis

### 주요 지표

**Primary Metric: Accuracy**
$$\text{Accuracy} = \frac{\text{Correct Answers}}{\text{Total Questions}} \times 100\%$$

**Confidence Interval (95% Bootstrap CI)**
```python
def bootstrap_ci(correct, n_bootstrap=1000):
    accuracies = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(correct, size=len(correct), replace=True)
        accuracies.append(np.mean(sample))
    return np.percentile(accuracies, [2.5, 97.5])
```

### 통계 분석

**ANOVA for Variance Decomposition**
- Nation effect vs Task effect 비교
- Effect size: η² (eta-squared)

**Pairwise Comparisons**
- Post-hoc t-tests with Bonferroni correction
- Cohen's d for effect size

**Model Consistency**
- Within-model Nation/Task variance ratio
- Binomial test for consistency across models

**Significance Testing**
- McNemar's test for paired comparisons
- p < 0.05 (α = 0.05)

---

## 4.6 Baselines

### 기준선 설정

**Random Baseline**
- 4지선다: 25%
- 5지선다: 20%
- 가중 평균: ~22%

**Human Baseline** (참고용)
- 각국 공무원 시험 합격선 기준
- Korea: ~60% (9급 기준)
- Japan: ~55%
- 비교 목적으로만 사용 (직접 비교 어려움)

---

## Section Summary

```
Evaluation Protocol 요약:

1. Task: Image-only multiple choice QA
2. Models: 23개 (9 closed + 14 open)
3. Metrics: Accuracy, 95% CI, Effect sizes
4. Diagnostic: 4-Track protocol for bottleneck analysis
5. Statistics: ANOVA, pairwise tests, consistency tests
```
