# Section 5: Main Results (한국어 초안)

---

## 5.1 Overall Performance

23개 VLM을 EuraGovExam 전체에서 평가한 결과를 Table 1에 제시한다.

### 주요 관찰

**Closed 모델이 Open 모델을 크게 앞섬:**
- 최고 성능: Gemini-2.5-pro (87.0%)
- Open 최고: Qwen2-VL-72B (44.6%)
- 성능 격차: 42.4%p

**최신 추론 모델의 강세:**
- o3 (84.3%), o4-mini (79.4%)가 상위권
- 추론 특화 모델이 시험 문제에 효과적

**Random baseline과의 비교:**
- 4-5지선다이므로 random = 20-25%
- 최하위 모델도 random보다는 높음 (Llama-3.2-11B: 12.7%는 예외)

### Table 1: Main Results

```
Table 1: Overall accuracy (%) on EuraGovExam. Models are sorted by overall performance.
Regional scores show significant variance within each model. Range indicates the 
performance gap between the easiest (Taiwan) and hardest (Japan) regions for each model.

| Model               | Overall | Taiwan | EU   | Korea | India | Japan | Range |
|---------------------|---------|--------|------|-------|-------|-------|-------|
| Gemini-2.5-pro      | 87.0    | 95.5   | 88.1 | 91.1  | 69.2  | 87.6  | 26.3  |
| o3                  | 84.3    | 93.7   | 84.5 | 90.1  | 68.6  | 82.4  | 25.1  |
| o4-mini             | 79.4    | 92.3   | 77.0 | 82.5  | 63.4  | 82.5  | 28.9  |
| Gemini-2.5-flash    | 68.3    | 92.7   | 83.3 | 67.7  | 62.3  | 51.5  | 41.2  |
| Claude-Sonnet-4     | 63.3    | 87.3   | 76.4 | 62.4  | 62.5  | 45.9  | 41.4  |
| GPT-4.1-mini        | 56.3    | 79.0   | 63.6 | 59.9  | 46.3  | 43.8  | 35.2  |
| GPT-4.1             | 54.7    | 72.6   | 66.4 | 54.2  | 48.1  | 48.1  | 24.5  |
| Qwen2-VL-72B        | 44.6    | 74.7   | 62.1 | 39.7  | 35.9  | 30.4  | 44.4  |
| GPT-4o              | 42.0    | 66.7   | 63.7 | 33.2  | 41.0  | 26.0  | 40.7  |
| InternVL2.5-38B     | 39.3    | 56.8   | 52.9 | 39.6  | 19.4  | 31.5  | 37.4  |
|---------------------|---------|--------|------|-------|-------|-------|-------|
| (Open models below) |         |        |      |       |       |       |       |
| Ovis2-32B           | 35.5    | 54.5   | 51.6 | 29.9  | 22.5  | 28.4  | 32.1  |
| Qwen2.5-VL-7B       | 32.3    | 47.0   | 45.9 | 29.5  | 26.3  | 21.9  | 25.1  |
| ...                 | ...     | ...    | ...  | ...   | ...   | ...   | ...   |

* Full results for all 23 models in Appendix C.
```

---

## 5.2 Core Finding: Regional Effect Dominates Task Effect

### 연구 질문

VLM 성능 차이의 주요 원인은 무엇인가?

- **가설 1 (Task difficulty)**: 수학이 어렵고 언어가 쉽다 → 과목 난이도가 성능 결정
- **가설 2 (Regional factors)**: 일본어가 어렵고 영어가 쉽다 → 지역/문자체계가 성능 결정

### 분석 방법

Two-way ANOVA를 사용하여 Nation과 Task가 성능 분산에 기여하는 정도를 분해:

$$\text{Accuracy}_{ijk} = \mu + \alpha_i^{\text{nation}} + \beta_j^{\text{task}} + \gamma_k^{\text{model}} + \epsilon_{ijk}$$

### 결과: 지역 효과가 과목 효과를 3.9배 압도

```
┌─────────────────────────────────────────────────────────────────┐
│                    CORE FINDING                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Factor      Variance    η²        F         p-value            │
│  ─────────────────────────────────────────────────────────      │
│  Nation      104.7      0.126     3.95      0.005**             │
│  Task         27.0      0.043     1.05      0.40 (n.s.)         │
│                                                                 │
│  ═══════════════════════════════════════════════════════════    │
│  Variance Ratio: 104.7 / 27.0 = 3.9×                            │
│  ═══════════════════════════════════════════════════════════    │
│                                                                 │
│  Interpretation:                                                │
│  • Nation effect is statistically significant (p < 0.01)        │
│  • Task effect is NOT significant (p = 0.40)                    │
│  • Nation explains 3.9× more variance than Task                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 효과 크기 해석

| Factor | η² | Interpretation |
|--------|-----|----------------|
| Nation | 0.126 | Medium effect (Cohen's guidelines: 0.06-0.14) |
| Task | 0.043 | Small effect (< 0.06) |

### 일관성 검증: 모델별 분석

개별 모델 내에서도 동일한 패턴이 관찰되는지 확인:

```
For each model, we computed:
- Nation variance: Var(accuracy across 5 nations)
- Task variance: Var(accuracy across 17 tasks)
- Ratio: Nation variance / Task variance

Results:
- 23개 모델 중 19개(82.6%)에서 Nation variance > Task variance
- Binomial test: p = 0.0013 (significantly more than chance)
- Mean ratio: 2.08
- Median ratio: 2.37
```

### Figure 2: Variance Decomposition

```
[Figure 설명]
좌측: Nation vs Task 분산 비교 막대그래프
- Nation bar가 Task bar보다 3.9배 높음
- 에러바로 95% CI 표시

우측: 모델별 Nation/Task ratio 분포
- 23개 점 중 19개가 ratio > 1 영역에 위치
- 빨간 점선: ratio = 1 (동일 효과)
```

---

## 5.3 Regional Performance Hierarchy

### 지역별 난이도 순위

```
┌─────────────────────────────────────────────────────────────────┐
│                 REGIONAL DIFFICULTY RANKING                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Hardest                                           Easiest      │
│  ◀─────────────────────────────────────────────────────────▶   │
│                                                                 │
│  Japan      India      Korea       EU        Taiwan             │
│  32.5%      32.6%      38.6%      49.9%      54.9%             │
│  ▓▓▓▓▓▓▓   ▓▓▓▓▓▓▓   ▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓▓▓ ▓▓▓▓▓▓▓▓▓▓▓▓     │
│                                                                 │
│  Performance Gap: 22.4 percentage points (Japan → Taiwan)       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 지역별 특성 분석

| Region | Mean Acc | Std | Script | Characteristics |
|--------|----------|-----|--------|-----------------|
| Japan | 32.5% | 23.4 | Kanji+Hiragana+Katakana | Most complex script system |
| India | 32.6% | 20.1 | Devanagari + Latin | Mixed scripts, reasoning challenge |
| Korea | 38.6% | 25.1 | Hangul | Syllabic script, OCR challenge |
| EU | 49.9% | 23.3 | Latin | Familiar to most models |
| Taiwan | 54.9% | 28.1 | Traditional Chinese | Best overall performance |

### 통계적 유의성: Pairwise Comparisons

```
Post-hoc pairwise t-tests with Bonferroni correction (α = 0.005):

| Comparison        | Mean Diff | t      | p        | Sig? |
|-------------------|-----------|--------|----------|------|
| Taiwan vs Japan   | +22.4%p   | 2.87   | 0.006    | *    |
| Taiwan vs India   | +22.4%p   | 3.04   | 0.004    | **   |
| EU vs Japan       | +17.3%p   | 2.46   | 0.018    |      |
| EU vs India       | +17.3%p   | 2.64   | 0.012    |      |
| Taiwan vs Korea   | +16.3%p   | 2.03   | 0.048    |      |

** p < 0.005 (Bonferroni-corrected significance)
*  p < 0.01
```

---

## 5.4 Evidence: Performance Gap is Not Due to Exam Difficulty

### 반론: "일본 시험이 원래 어려운 거 아니야?"

이 반론을 검증하기 위해, **동일한 시험에 대해 모델 간 성능 격차**를 분석했다.

### 핵심 증거: 같은 시험인데 모델마다 격차가 다름

```
┌─────────────────────────────────────────────────────────────────┐
│  If Japanese exams were "inherently harder":                    │
│  → All models should show similar performance drops             │
│                                                                 │
│  But we observe:                                                │
│                                                                 │
│  Model            Taiwan    Japan     Gap                       │
│  ─────────────────────────────────────────                      │
│  GPT-4o           66.7%     26.0%     -40.7%p  ← Huge drop!    │
│  Gemini-2.5-pro   95.5%     87.6%     - 7.9%p  ← Small drop    │
│                                                                 │
│  Same exam, different models, different gaps!                   │
│  → The gap is about MODEL CAPABILITY, not exam difficulty       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 모델별 지역 격차 분포

```
Regional Gap (Taiwan - Japan) by Model:

| Model               | Taiwan | Japan | Gap    |
|---------------------|--------|-------|--------|
| Gemini-2.5-flash-lite| 73.3% | 7.1%  | 66.2%p | ← Largest
| GPT-4o              | 66.7%  | 26.0% | 40.7%p |
| Claude-Sonnet-4     | 87.3%  | 45.9% | 41.4%p |
| Gemini-2.5-flash    | 92.7%  | 51.5% | 41.2%p |
| Qwen2-VL-72B        | 74.7%  | 30.4% | 44.3%p |
| ...                 | ...    | ...   | ...    |
| Gemini-2.5-pro      | 95.5%  | 87.6% | 7.9%p  | ← Smallest
| o3                  | 93.7%  | 82.4% | 11.3%p |
| o4-mini             | 92.3%  | 82.5% | 9.8%p  |

Observation:
- Gap ranges from 7.9%p to 66.2%p
- If exams were equally hard, all gaps should be similar
- → Performance gap reflects MODEL's language capability
```

### 한국 시험 사례: 결정적 증거

```
한국 공무원 시험 성능:

GPT-4o:        33.2%  (랜덤 수준)
Gemini-2.5-pro: 91.1%  (거의 완벽)

차이: 57.9 percentage points!

→ 같은 한국 시험인데:
  - GPT-4o는 처참하게 실패 (33%)
  - Gemini는 잘 풂 (91%)
  
→ 한국 시험이 어려운 게 아니라, GPT-4o가 한글을 못 읽는 것!
```

---

## 5.5 Model Type Analysis

### Closed vs Open Models

```
Nation/Task Variance Ratio by Model Type:

| Model Type    | N  | Mean Ratio | Median Ratio |
|---------------|-----|------------|--------------|
| Closed models | 9   | 2.45       | 2.58         |
| Open models   | 14  | 1.85       | 2.23         |

t-test: t = 1.14, p = 0.27 (not significant)

→ Both closed and open models show Nation > Task pattern
→ The regional effect is universal across model types
```

### 성능 수준별 분석

```
Does the regional effect persist across performance levels?

| Performance Tier | Models | Mean Nation/Task Ratio |
|------------------|--------|------------------------|
| High (>60%)      | 5      | 2.38                   |
| Medium (30-60%)  | 12     | 2.15                   |
| Low (<30%)       | 6      | 1.72                   |

→ Regional effect is consistent across all performance levels
→ Even the best models show Nation > Task
```

---

## 5.6 Summary of Main Findings

```
┌─────────────────────────────────────────────────────────────────┐
│                    KEY TAKEAWAYS                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. REGIONAL EFFECT DOMINATES                                   │
│     • Nation explains 3.9× more variance than Task              │
│     • η²(Nation) = 0.126 vs η²(Task) = 0.043                   │
│     • Statistically significant: p = 0.005                      │
│                                                                 │
│  2. CONSISTENT ACROSS MODELS                                    │
│     • 82.6% of models (19/23) show Nation > Task               │
│     • Pattern holds for both closed and open models             │
│     • Binomial test: p = 0.0013                                 │
│                                                                 │
│  3. LARGE PERFORMANCE GAPS                                      │
│     • Japan (32.5%) vs Taiwan (54.9%): 22.4%p gap              │
│     • GPT-4o: EU 63.7% vs Japan 26.0%: 37.7%p gap              │
│                                                                 │
│  4. NOT DUE TO EXAM DIFFICULTY                                  │
│     • Same exam, different models, different gaps               │
│     • GPT-4o drops 40.7%p, Gemini drops only 7.9%p              │
│     • → Gap reflects model capability, not exam difficulty      │
│                                                                 │
│  IMPLICATION:                                                   │
│  VLM reliability varies dramatically by region.                 │
│  Region-specific validation is REQUIRED before deployment.      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Figure/Table 목록 (이 섹션)

| # | Type | Content | Priority |
|---|------|---------|----------|
| Table 1 | Results | Main results (23 models × 5 regions) | HIGH |
| Table 2 | Stats | ANOVA results (Nation vs Task) | HIGH |
| Figure 2 | Bar chart | Variance decomposition | HIGH |
| Figure 3 | Bar chart | Regional difficulty ranking | HIGH |
| Figure 4 | Scatter | Model-specific regional gaps | MEDIUM |
