# Section 1: Introduction (한국어 초안)

---

## Opening Paragraph (Hook)

Vision-Language Model(VLM)의 성능이 지난 2년간 급격히 향상되었다. Gemini-2.5-pro는 MMMU에서 90%를 넘어섰고, GPT-4o는 미국 의사 면허 시험과 변호사 시험을 합격 수준으로 통과한다. 이러한 발전에 힘입어, 각국 정부는 행정 문서 처리, 민원 자동화, 시험 채점 등 고부담(high-stakes) 업무에 AI 도입을 적극 검토하고 있다. 한국 행정안전부는 AI 민원 상담 시스템을, 일본 デジタル庁는 행정 문서 자동 처리를, EU는 AI Act 하에서 공공서비스 AI 규제 프레임워크를 수립 중이다.

그러나 정부 시스템에 VLM을 배포하기 전에, 우리는 핵심적인 질문에 답해야 한다:

> **"영어 벤치마크에서 우수한 모델이 한국어, 일본어, 힌디어 공문서에서도 신뢰할 수 있는가?"**

본 연구는 이 질문에 대해 **"아니오"**라고 답한다.

---

## 1.1 The Rise of VLMs and Government AI Adoption

### VLM 성능 발전 타임라인

```
┌─────────────────────────────────────────────────────────────────┐
│                 VLM PERFORMANCE TIMELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  2023                              2025                         │
│  ────────                          ────────                     │
│  GPT-4V: MMMU 56%      →          Gemini-2.5: MMMU 90%+        │
│  Medical exam: Fail    →          Medical exam: Top 10%        │
│  Bar exam: Borderline  →          Bar exam: Pass               │
│                                                                 │
│  "AI is approaching expert-level performance"                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 정부 AI 도입 추세

| Country | Initiative | Status |
|---------|------------|--------|
| Korea | AI 민원 상담 시스템 | Pilot testing |
| Japan | 행정 문서 자동 처리 | デジタル庁 주관 실증 |
| EU | AI Act 공공서비스 규제 | 2024년 발효 |
| US | Federal AI use guidelines | OMB Memo 2024 |
| Singapore | AI in government services | Deployed |

### 핵심 메시지

정부는 VLM이 "충분히 똑똑해졌다"고 판단하고 도입을 서두르고 있다. 그러나 이 판단은 주로 영어 벤치마크에 기반한다. 다른 언어와 문자체계에서도 동일한 신뢰성을 기대할 수 있는가?

---

## 1.2 The Critical Question: Can We Trust VLMs Across Languages?

### 기존 벤치마크의 한계

```
┌─────────────────────────────────────────────────────────────────┐
│              LIMITATIONS OF EXISTING BENCHMARKS                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Benchmark    Language   Image Type      Knowledge Type         │
│  ─────────────────────────────────────────────────────────      │
│  MMMU         English    Clean renders   Universal (STEM)       │
│  MathVista    English    Synthetic       Mathematical           │
│  EXAMS-V      Multi      Rendered        School curriculum      │
│                                                                 │
│  Common issues:                                                 │
│  • English-centric evaluation                                   │
│  • Synthetically rendered images (not real scans)              │
│  • Universal knowledge (math, science) that transfers          │
│  • No jurisdiction-specific knowledge testing                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### "범용 AI" 가정의 위험

많은 사람들이 다음과 같이 가정한다:
- "MMMU 90% 모델은 어떤 문서든 잘 처리할 것이다"
- "영어에서 잘 작동하면 한국어에서도 비슷할 것이다"
- "추론 능력이 좋으면 언어와 무관하게 문제를 풀 것이다"

**이 가정이 틀렸다면, 정부 시스템에서 심각한 문제가 발생할 수 있다:**
- 민원 오처리로 인한 시민 피해
- 자동화된 심사의 불공정성
- 법적 책임 문제

### 본 연구의 질문

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESEARCH QUESTION                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  "A model that scores 90% on MMMU—                              │
│   can it reliably process documents in Korean, Japanese,        │
│   or Hindi?"                                                    │
│                                                                 │
│  → This is the question EuraGovExam is designed to answer.      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1.3 Our Finding: VLM Reliability Varies Dramatically by Region

### 핵심 발견 요약

23개 최신 VLM을 5개 지역(한국, 일본, 대만, 인도, EU)의 실제 공무원 시험 8,000+개에서 평가한 결과:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALARMING FINDING                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Regional effect is 3.9× larger than task effect                │
│                                                                 │
│  • η²(Region) = 0.126 (medium effect)                          │
│  • η²(Task)   = 0.043 (small effect)                           │
│  • p < 0.01 (statistically significant)                        │
│                                                                 │
│  This means:                                                    │
│  → WHERE the exam is from matters 3.9× more than               │
│    WHAT subject it tests                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### GPT-4o 사례: 가장 극적인 예시

```
┌─────────────────────────────────────────────────────────────────┐
│                 GPT-4o PERFORMANCE COLLAPSE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Region      Accuracy    Interpretation                         │
│  ───────────────────────────────────────────────                │
│  Taiwan      66.7%       Reasonable                             │
│  EU          63.7%       Reasonable                             │
│  India       41.0%       Below average                          │
│  Korea       33.2%       Near random (25%)                      │
│  Japan       26.0%       Essentially random!                    │
│                                                                 │
│  ════════════════════════════════════════════════════════════   │
│  Same model. Same architecture. Same weights.                   │
│  Performance drops by 2.5× just by changing the region.         │
│  ════════════════════════════════════════════════════════════   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Figure 1 설명 (Introduction의 핵심 Figure)

```
[Figure 1: VLM reliability varies dramatically by region]

(a) GPT-4o performance by region (bar chart)
    - 5개 막대: Taiwan (66.7%), EU (63.7%), India (41.0%), Korea (33.2%), Japan (26.0%)
    - 빨간 점선: Random baseline (25%)
    - Japan 막대에 "Near random!" 표시

(b) Regional gap varies by model (scatter plot)
    - X축: Taiwan accuracy
    - Y축: Japan accuracy
    - 대각선: Equal performance line
    - GPT-4o: 대각선에서 멀리 떨어짐
    - Gemini-2.5-pro: 대각선 근처

Caption: VLM performance is highly inconsistent across regions. 
(a) GPT-4o achieves 63.7% on EU documents but only 26.0% on Japanese 
documents—a 2.5× performance gap. (b) This gap varies by model: 
GPT-4o drops 40.7 percentage points from Taiwan to Japan, while 
Gemini-2.5-pro drops only 7.9 points on the same exams.
```

### 일관성: 82.6% 모델에서 관찰

```
This is not an isolated case:

• 23 models evaluated
• 19 models (82.6%) show Region > Task variance
• Binomial test: p = 0.0013

→ The regional performance gap is a SYSTEMATIC phenomenon,
   not an artifact of specific models.
```

---

## 1.4 Why Civil Service Exams?

### 공무원 시험의 특성

| Property | Academic Exams | Civil Service Exams |
|----------|----------------|---------------------|
| Knowledge type | Universal (math, science) | Jurisdiction-specific (law, admin) |
| Image source | Rendered from text | Real scanned documents |
| Ground truth | Crowd-sourced | Official government answer keys |
| Stakes | Educational | Employment (high-stakes) |
| Transferability | Translatable | NOT translatable |

### 왜 공무원 시험이 정부 AI 검증에 적합한가?

```
┌─────────────────────────────────────────────────────────────────┐
│         WHY CIVIL SERVICE EXAMS ARE IDEAL                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. NATURALLY OCCURRING EVALUATION                              │
│     • Not artificially constructed for AI                       │
│     • Real high-stakes assessment (government employment)       │
│     • Quality guaranteed by official review process             │
│                                                                 │
│  2. OFFICIAL GROUND TRUTH                                       │
│     • Government-published answer keys                          │
│     • No crowd-sourcing noise                                   │
│     • Legally verified correctness                              │
│                                                                 │
│  3. JURISDICTION-SPECIFIC KNOWLEDGE                             │
│     • Korean law ≠ Japanese law ≠ EU law                        │
│     • Cannot be solved by translation alone                     │
│     • Tests TRUE regional understanding                         │
│                                                                 │
│  4. REAL DOCUMENT FORMAT                                        │
│     • Scanned PDFs with noise and artifacts                     │
│     • Complex layouts (tables, multi-column)                    │
│     • Tests OCR + understanding + reasoning                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### EXAMS-V와의 차별화

```
EXAMS-V (Das et al., 2024):
• 21,000 school exam questions
• Universal knowledge (can be translated)
• Rendered images (not real scans)

EuraGovExam (Ours):
• 8,000+ civil service exam questions
• Jurisdiction-specific knowledge (cannot be translated)
• Real scanned documents

Key difference: 
EXAMS-V tests "Can you do math in Korean?"
EuraGovExam tests "Do you understand Korean administrative law?"
```

---

## 1.5 Contributions

### 본 연구의 기여

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTRIBUTIONS                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. RELIABILITY WARNING (Primary Contribution)                  │
│     ─────────────────────────────────────────                   │
│     "VLMs cannot be trusted uniformly across languages.         │
│      Region-specific validation is REQUIRED before              │
│      government deployment."                                    │
│                                                                 │
│     Evidence:                                                   │
│     • Regional effect 3.9× larger than task effect              │
│     • Consistent across 82.6% of 23 models                      │
│     • GPT-4o: 63.7% (EU) vs 26.0% (Japan)                       │
│                                                                 │
│  2. EuraGovExam BENCHMARK                                       │
│     ─────────────────────────                                   │
│     • 8,000+ real civil service exam questions                  │
│     • 5 regions × 4 writing systems × 17 subjects               │
│     • Official answer keys                                      │
│     • Designed for government AI validation                     │
│                                                                 │
│  3. DIAGNOSTIC PROTOCOL                                         │
│     ───────────────────────                                     │
│     • 4-Track evaluation to decompose failures                  │
│     • Separates OCR bottleneck vs reasoning bottleneck          │
│     • Provides actionable improvement directions                │
│                                                                 │
│  4. REGIONAL GUIDELINES                                         │
│     ────────────────────────                                    │
│     • Japan/Korea: Improve OCR for complex scripts              │
│     • India: Improve reasoning capability                       │
│     • EU: Current models adequate                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1.6 Paper Organization

The remainder of this paper is organized as follows:

- **Section 2** reviews related work on multimodal benchmarks and VLM evaluation
- **Section 3** describes the EuraGovExam dataset construction and statistics
- **Section 4** presents our evaluation protocol and experimental setup
- **Section 5** reports main results showing regional effects dominate task effects
- **Section 6** provides diagnostic analysis of regional bottleneck patterns
- **Section 7** discusses implications for government AI deployment
- **Section 8** concludes with limitations and future directions

---

## Key Sentences for This Section (영어 버전 핵심 문장)

### Opening Hook
> "With VLMs achieving expert-level performance on professional examinations, governments worldwide are actively exploring AI deployment for administrative document processing. But can a model that passes the US bar exam reliably process Korean government documents?"

### Problem Statement
> "Existing benchmarks—predominantly English-centric and based on synthetically rendered images—fail to answer this critical question. MMMU scores do not predict performance on real multilingual government documents."

### Key Finding
> "Our experiments reveal an alarming finding: regional/jurisdictional effects dominate task/subject effects by a 3.9× variance ratio (η²=0.126 vs 0.043, p<0.01). GPT-4o achieves 63.7% on EU documents but plummets to 26.0% on Japanese documents—barely above random chance."

### Implication
> "These results challenge the assumption of 'universal AI capability' and demonstrate that region-specific validation is not optional but essential before deploying VLMs in government systems."

### Contribution Statement
> "We introduce EuraGovExam, a benchmark of 8,000+ authentic civil service examination questions from five Eurasian regions, designed specifically to test government AI readiness across languages and jurisdictions."

---

## Figure/Table 목록 (이 섹션)

| # | Type | Content | Priority |
|---|------|---------|----------|
| Figure 1 | Composite | (a) GPT-4o regional collapse, (b) Model-specific gaps | HIGH |
| Table 1 | Comparison | Academic exams vs Civil service exams | MEDIUM |
