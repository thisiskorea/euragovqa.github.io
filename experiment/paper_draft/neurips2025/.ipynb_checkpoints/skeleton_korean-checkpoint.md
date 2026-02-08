# EuraGovExam: Can We Trust VLMs for Government Document Processing?

## 논문 스켈레톤 (한국어 초안)

---

# Title
**EuraGovExam: Can We Trust VLMs for Government Document Processing?**

**Alternative**: Mind the Gap: VLM Reliability Varies 3.9× Across Languages in Civil Service Exams

---

# Abstract (250 words)

## 한국어 초안

**[배경 - AI 도입 추세]**
Vision-Language Model(VLM)의 성능이 급격히 향상되면서, 각국 정부는 행정 문서 처리, 민원 자동화, 시험 채점 등 고부담(high-stakes) 업무에 AI 도입을 적극 검토하고 있다.

**[핵심 질문]**
그러나 정부 시스템에 VLM을 배포하기 전에, 우리는 핵심적인 질문에 답해야 한다: "영어 벤치마크에서 우수한 모델이 다른 나라의 공문서에서도 신뢰할 수 있는가?"

**[데이터셋]**
이 질문에 답하기 위해, 우리는 EuraGovExam을 제안한다—한국, 일본, 대만, 인도, EU 5개 지역의 실제 공무원 시험 8,000+개로 구성된 다국어 멀티모달 벤치마크이다. 공무원 시험은 관할권별 특수 지식을 실제 스캔 문서 형태로 평가하므로, 정부 AI 신뢰성 검증에 이상적인 테스트베드이다.

**[핵심 발견]**
23개 최신 VLM을 평가한 결과, 우려스러운 발견을 했다: 지역/언어에 따른 성능 격차가 과목 난이도 격차보다 3.9배 더 크다 (η²=0.126 vs 0.043, p<0.01). GPT-4o는 EU 문서에서 63.7%를 달성하지만 일본 문서에서는 26.0%로 추락한다—거의 랜덤 수준이다. 이러한 패턴은 23개 모델 중 82.6%에서 일관되게 관찰된다.

**[시사점]**
이 결과는 VLM의 "범용 지능"에 대한 과신을 경고하며, 정부가 AI를 도입하기 전에 대상 지역/언어에 특화된 검증이 필수임을 시사한다.

---

# 1. Introduction (2 pages)

## 1.1 VLM의 급격한 발전과 정부 도입 추세

### 핵심 포인트
- VLM 성능이 2년 만에 급격히 향상 (MMMU 56% → 90%+)
- 의사시험, 변호사시험 등 전문가 시험 합격 수준 도달
- 이에 따라 각국 정부가 AI 도입을 적극 검토

### 들어갈 내용
- 성능 발전 타임라인 (2023 → 2025)
- 정부 AI 도입 사례/계획 (한국, 일본, EU, 미국)
- "AI가 충분히 똑똑해졌다"는 인식 확산

### 핵심 문장
> "With VLMs achieving expert-level performance on professional exams, governments worldwide are actively exploring AI deployment for administrative document processing, citizen services, and examination grading."

---

## 1.2 핵심 질문: 정말 믿을 수 있는가?

### 핵심 포인트
- 기존 벤치마크의 한계: 영어 중심, 깨끗한 이미지
- MMMU 90% ≠ 한국 공문서 처리 능력
- 신뢰성 검증 없이 배포하면 위험

### 들어갈 내용
- 기존 벤치마크 목록과 한계점 (MMMU, MathVista, EXAMS-V)
- "범용 AI"라는 가정의 위험성
- 정부 시스템에서의 실패 비용 (민원 오처리, 법적 책임)

### 핵심 문장
> "Before deploying VLMs in high-stakes government systems, we must answer a critical question: Can a model that excels on English benchmarks reliably process documents in Korean, Japanese, or Hindi?"

---

## 1.3 우리의 발견: 신뢰할 수 없다

### 핵심 포인트
- 지역/언어에 따라 성능이 3.9배 차이
- GPT-4o: EU 63.7% vs Japan 26.0% (2.5배 차이)
- 같은 모델인데 지역만 바뀌면 랜덤 수준으로 추락

### 들어갈 내용
- 핵심 통계 (3.9× variance ratio, η², p-value)
- GPT-4o 사례 (가장 극적인 예시)
- 82.6% 모델에서 일관적 패턴

### Figure 1 아이디어
- 5개 지역에서 GPT-4o 성능 막대그래프
- EU (63.7%) → Japan (26.0%) 급락 강조

### 핵심 문장
> "Our experiments reveal an alarming finding: GPT-4o achieves 63.7% on EU documents but plummets to 26.0% on Japanese documents—barely above random chance. This is not an isolated case; 82.6% of the 23 models we evaluated show similar regional performance collapse."

---

## 1.4 왜 공무원 시험인가?

### 핵심 포인트
- 자연 발생 평가 도구 (인위적으로 만든 게 아님)
- 공식 정답 존재 (정부 발표)
- 관할권별 특수 지식 필요 (번역으로 해결 안 됨)
- 실제 스캔 문서 형태 (OCR + 이해 + 추론)

### 들어갈 내용
- EXAMS-V vs EuraGovExam 비교표
- 학교 시험 vs 공무원 시험 차이
- 왜 공무원 시험이 정부 AI 신뢰성 검증에 적합한지

### 핵심 문장
> "Civil service examinations are ideal for testing government AI readiness: they are naturally-occurring evaluation instruments with official ground truth, requiring jurisdiction-specific knowledge embedded in real scanned documents."

---

## 1.5 Contributions

### 4가지 기여

1. **신뢰성 경고 (Primary)**
   - "VLM을 정부 시스템에 도입하기 전, 대상 지역/언어에서 반드시 검증해야 한다"
   - 증거: 지역 효과가 과목 효과를 3.9배 압도

2. **EuraGovExam 벤치마크**
   - 5개 지역, 4개 문자체계, 8,000+ 문항
   - 정부 AI 도입 전 검증용 도구

3. **진단 프로토콜**
   - 4-Track 분석으로 실패 원인 분해
   - OCR 병목 vs 추론 병목 구분

4. **지역별 가이드라인**
   - 일본/한국: OCR 강화 필요
   - 인도: 추론/지식 강화 필요
   - EU: 현재 수준 OK

---

# 2. Related Work (1 page)

## 2.1 Exam and Knowledge Benchmarks

### 다룰 벤치마크
- MMLU (Hendrycks et al., 2021): 텍스트 전용, 영어
- MMMU (Yue et al., 2024): 멀티모달, 대학 수준, 깨끗한 이미지
- EXAMS-V (Das et al., 2024): 다국어, 21K, 학교 시험
- MathVista (Lu et al., 2024): 수학 시각 추론

### 차별화 포인트
- 기존: 보편적 지식 (수학, 과학)
- 우리: 관할권별 특수 지식 (법률, 행정)
- 기존: 렌더링된 깨끗한 이미지
- 우리: 실제 스캔 문서

---

## 2.2 Document Understanding Benchmarks

### 다룰 벤치마크
- DocVQA (Mathew et al., 2021)
- ChartQA (Masry et al., 2022)
- TextVQA (Singh et al., 2019)
- OCRBench (Liu et al., 2024)

### 차별화 포인트
- 기존: 영어 중심, 단일 도메인
- 우리: 다국어, 다양한 문자체계, 정부 문서

---

## 2.3 VLM Reliability and Robustness Studies

### 다룰 연구
- LogicOCR 류의 진단 연구
- Perception bottleneck 분석
- Multilingual VLM evaluation

### 포지셔닝
- 기존 연구는 "방법론" 제안
- 우리는 기존 방법론의 "체계적 적용" + "새로운 발견"
- 핵심 기여는 방법론이 아니라 실증적 발견

---

# 3. EuraGovExam Dataset (2 pages)

## 3.1 Design Rationale

### 왜 이렇게 설계했나
- 정부 AI 신뢰성 검증이 목적
- 실제 문서 형태 유지 (스캔 노이즈, 복잡한 레이아웃)
- 다양한 문자체계 포함
- 관할권별 지식 필요하도록

---

## 3.2 Data Sources

### Table: 지역별 데이터 출처

| Region | Source | Language | Script | Questions |
|--------|--------|----------|--------|-----------|
| Korea | 인사혁신처 | Korean | Hangul | ~2,000 |
| Japan | 人事院 | Japanese | Kanji+Kana | ~1,500 |
| Taiwan | 考選部 | Chinese | Traditional | ~1,500 |
| India | UPSC | Hindi/English | Devanagari/Latin | ~1,500 |
| EU | EPSO | EN/FR/DE | Latin | ~1,500 |

---

## 3.3 Data Collection Pipeline

### 단계별 설명
1. 원본 수집 (정부 웹사이트, PDF, 스캔)
2. 전처리 (페이지 분리, 문항 크롭, 해상도 정규화)
3. 정답 레이블링 (공식 정답표 매칭)
4. 메타데이터 부착 (nation, task, year, difficulty)
5. 품질 검증 (이중 검증, 오류율 <0.1%)

### Figure: Pipeline Diagram

---

## 3.4 Dataset Statistics

### 핵심 통계
- 총 문항: 8,000+
- 지역: 5개
- 과목: 17개
- 문자체계: 4개
- 연도 범위: 2015-2024

### Figure: 과목 분포, 지역 분포 파이차트

---

## 3.5 Data Quality and Contamination Analysis

### 품질 보장
- Near-duplicate 탐지 (perceptual hash)
- PII 제거 검증
- 정답 이중 검증

### Contamination 분석
- 연도별 성능 비교 (2023+ vs 이전)
- Clean subset 정의 (temporal clean)

---

# 4. Evaluation Protocol (1 page)

## 4.1 Task Definition

### Image-Only Setting
- 입력: 시험 이미지만
- 출력: 정답 번호 (1-5)
- 제약: OCR 도구 사용 금지

---

## 4.2 Evaluated Models

### Table: 23개 모델 목록

| Type | Models |
|------|--------|
| Closed (9) | Gemini-2.5-pro, o3, o4-mini, GPT-4o, GPT-4.1, GPT-4.1-mini, Claude-Sonnet-4, Gemini-2.5-flash, Gemini-2.5-flash-lite |
| Open (14) | Qwen2-VL (2B/7B/72B), InternVL2.5-38B, LLaVA-1.5 (7B/13B), Llama-3.2-11B-Vision, Ovis2 (8B/16B/32B), Phi-3.5-vision, etc. |

---

## 4.3 Diagnostic 4-Track Protocol

### 4개 Track 설명

1. **Track A: Image Only**
   - VLM에게 이미지만 제공
   - 순수 시각+추론 능력 측정

2. **Track B1: OCR → LLM**
   - OCR로 텍스트 추출 → LLM에게 텍스트만 제공
   - 읽기 병목 제거 시 성능

3. **Track B2: Oracle Text → LLM**
   - 인간이 정확히 타이핑한 텍스트 → LLM
   - 완벽한 읽기 가정 시 성능 상한선

4. **Track C: Image + OCR Text**
   - VLM에게 이미지 + OCR 텍스트 함께 제공
   - 멀티모달 시너지 측정

### Figure: 4-Track Protocol Diagram

---

## 4.4 Metrics

- Accuracy (주요 지표)
- 95% Bootstrap Confidence Interval
- Cohen's d (effect size)
- McNemar's test (paired comparison)

---

# 5. Main Results (2 pages) ⭐ 핵심 섹션

## 5.1 Overall Performance

### Table 1: Main Results

| Model | Overall | Taiwan | EU | Korea | India | Japan | Range |
|-------|---------|--------|-----|-------|-------|-------|-------|
| Gemini-2.5-pro | 87.0 | 95.5 | 88.1 | 91.1 | 69.2 | 87.6 | 26.3 |
| o3 | 84.3 | 93.7 | 84.5 | 90.1 | 68.6 | 82.4 | 25.1 |
| ... | ... | ... | ... | ... | ... | ... | ... |

### 주요 발견
- Top 모델: Gemini-2.5-pro (87.0%)
- Open 최고: Qwen2-VL-72B (44.6%)
- Random baseline: 20-25%

---

## 5.2 Core Finding: Region >> Task ⭐⭐⭐

### 연구 질문
"VLM 성능 차이는 과목(수학 vs 법학)에서 오는가, 지역(한국 vs 일본)에서 오는가?"

### 결과

**지역 효과가 과목 효과를 3.9배 압도!**

| Factor | Variance | η² | F | p |
|--------|----------|-----|---|---|
| Nation | 104.7 | 0.126 (medium) | 3.95 | 0.005** |
| Task | 27.0 | 0.043 (small) | 1.05 | 0.40 |

- Ratio: 104.7 / 27.0 = **3.9×**
- 23개 모델 중 19개(82.6%)에서 Nation > Task
- Binomial test: p = 0.0013

### Figure 2: Variance Decomposition (Nation vs Task)

---

## 5.3 Regional Performance Gap

### 지역별 난이도

| Rank | Region | Accuracy | Interpretation |
|------|--------|----------|----------------|
| 1 (Hardest) | Japan | 32.5% | Complex script (Kanji+Kana) |
| 2 | India | 32.6% | Reasoning bottleneck |
| 3 | Korea | 38.6% | OCR bottleneck |
| 4 | EU | 49.9% | Latin script advantage |
| 5 (Easiest) | Taiwan | 54.9% | Best overall |

- Performance gap: 22.4 percentage points (Japan → Taiwan)

### Figure 3: Nation Difficulty Ranking Bar Plot

---

## 5.4 Evidence: Not Just Harder Exams

### 반론 검증: "일본 시험이 원래 어려운 거 아니야?"

**같은 시험인데 모델마다 격차가 다름:**

| Model | Taiwan | Japan | Gap |
|-------|--------|-------|-----|
| GPT-4o | 66.7% | 26.0% | **40.7%p** |
| Gemini-2.5-pro | 95.5% | 87.6% | **7.9%p** |

- 만약 일본 시험이 "원래 어려운 것"이라면, 두 모델 다 비슷한 비율로 떨어져야 함
- 하지만 GPT-4o는 40%p 떨어지고, Gemini는 8%p만 떨어짐
- **→ 시험 난이도가 아니라 "모델의 언어 처리 능력" 차이!**

### Figure 4: Model-specific Regional Gaps (GPT-4o vs Gemini)

---

# 6. Diagnostic Analysis (2 pages)

## 6.1 Decomposing the Regional Gap

### 질문: 왜 지역마다 성능이 다른가?

### 가설들
1. OCR/읽기 능력 차이
2. 추론 능력 차이
3. 관할권별 지식 차이

### 4-Track 분석으로 분해

---

## 6.2 Regional Bottleneck Patterns

### Table: 지역별 병목 유형

| Region | Bottleneck Type | Evidence | Recommendation |
|--------|-----------------|----------|----------------|
| Japan | OCR bottleneck | Track B >> Track A | Improve Japanese OCR |
| Korea | OCR bottleneck | Track B > Track A | Improve Korean OCR |
| India | Reasoning bottleneck | Track A ≈ B ≈ C | Improve reasoning |
| EU | Normal synergy | Track C > B > A | Current level OK |
| Taiwan | Best overall | High across tracks | - |

---

## 6.3 Model-Specific Insights

### Gemini vs GPT 비교

**Gemini 계열: 아시아 언어 강함**
- Japan: 87.6% (GPT-4o 대비 +61.6%p)
- Korea: 91.1% (GPT-4o 대비 +57.9%p)

**GPT 계열: 아시아 언어 약함**
- Japan: 26.0%
- Korea: 33.2%

### 원인 추정
- 학습 데이터 구성 차이
- 내장 OCR 품질 차이
- 멀티모달 인코더 설계 차이

---

## 6.4 Failure Case Analysis

### 대표적 실패 사례 분석

**Case 1: 일본 법률 문제**
- 문제: 복잡한 한자 + 법률 용어
- GPT-4o: 한자 오인식 → 오답
- Gemini: 정확히 인식 → 정답

**Case 2: 한국 행정 문제**
- 문제: 표 형태 + 한글
- 대부분 모델: 표 구조 파악 실패

### Figure: Failure Case Examples

---

# 7. Discussion (1 page)

## 7.1 Implications for Government AI Deployment

### 핵심 메시지
- VLM의 "범용 지능"을 과신하면 안 됨
- 정부 AI 도입 전 대상 지역/언어에서 반드시 검증 필요
- 영어 벤치마크 성능 ≠ 다른 언어 성능

### 실용적 권고
1. 배포 전 대상 지역 데이터로 별도 평가
2. 지역별 fine-tuning 또는 모델 선택 최적화
3. 고부담 업무는 human-in-the-loop 유지

---

## 7.2 Why Do Regional Gaps Exist?

### 원인 분석
1. **학습 데이터 불균형**: 영어 >> 아시아 언어
2. **OCR 품질 차이**: 라틴 문자 >> 복잡한 문자체계
3. **관할권 지식 부족**: 미국법 지식 >> 한국법 지식

### 개선 방향
- 다국어 학습 데이터 확충
- 아시아 언어 OCR 품질 개선
- 지역별 fine-tuning 데이터셋 구축

---

## 7.3 Limitations

### 한계점
1. **Coverage**: 5개 지역만 포함 (동남아, 중동, 아프리카 미포함)
2. **Contamination**: 일부 오래된 시험은 학습 데이터에 포함됐을 가능성
3. **OCR Dependency**: 우리 OCR 엔진 품질에 의존
4. **Causality**: 관찰 연구이므로 인과관계 확정 어려움

---

# 8. Conclusion (0.5 page)

## 핵심 요약

1. **문제 제기**: VLM이 정부 시스템에 도입되려 하는데, 신뢰할 수 있는가?

2. **벤치마크**: EuraGovExam—5개 지역, 8,000+ 공무원 시험

3. **핵심 발견**: 
   - 지역 효과가 과목 효과를 3.9배 압도
   - GPT-4o는 EU 63.7% vs Japan 26.0%
   - 82.6% 모델에서 일관적 패턴

4. **시사점**: 
   - VLM의 "범용 지능" 과신 금지
   - 정부 AI 도입 전 지역별 검증 필수
   - EuraGovExam = 다국어 정부 AI 신뢰성 검증 도구

## 한 줄 결론

> "정부가 AI를 믿고 쓰려 하지만, 지역/언어에 따라 성능이 3.9배 차이나므로 대상 지역에서 반드시 검증해야 한다."

---

# Appendix

## A. Datasheet for EuraGovExam
(이미 작성됨: datasheet.md)

## B. Croissant Metadata
(이미 작성됨: croissant_metadata.json)

## C. Full Model Results
- 23개 모델 × 5개 지역 × 17개 과목 전체 결과 테이블

## D. Additional Experiments
- Track별 상세 결과
- Robustness 분석 (해상도 변화, 노이즈 추가)

## E. Evaluation Code
- GitHub 링크
- 재현성 가이드

---

# Figure/Table 목록

| # | Type | Content | Status |
|---|------|---------|--------|
| Fig 1 | Bar plot | GPT-4o regional performance collapse | TODO |
| Fig 2 | Pie/Bar | Variance decomposition (Nation vs Task) | Done |
| Fig 3 | Bar plot | Nation difficulty ranking | Done |
| Fig 4 | Line plot | Model-specific regional gaps | TODO |
| Fig 5 | Diagram | 4-Track Protocol | TODO |
| Fig 6 | Heatmap | Model × Nation performance | Done |
| Fig 7 | Examples | Failure case images | TODO |
| Table 1 | Results | Main results (Top 10 models) | Done |
| Table 2 | Stats | ANOVA results | Done |
| Table 3 | Data | Dataset statistics | TODO |
| Table 4 | Analysis | Regional bottleneck patterns | TODO |

---

# 작성 우선순위

1. **Section 5 (Main Results)** - 핵심 발견, 가장 중요
2. **Section 1 (Introduction)** - 스토리 설정
3. **Abstract** - 전체 요약
4. **Section 6 (Diagnostic Analysis)** - 원인 분석
5. **Section 3 (Dataset)** - 데이터 설명
6. **Section 4 (Protocol)** - 방법론
7. **Section 2 (Related Work)** - 선행연구
8. **Section 7-8 (Discussion, Conclusion)** - 마무리
