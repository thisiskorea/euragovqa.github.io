# Section 6: Diagnostic Analysis (한국어 초안)

---

## 6.1 Research Questions

Main Results에서 지역 효과가 과목 효과를 3.9배 압도한다는 것을 확인했다. 이 섹션에서는 **왜** 이러한 현상이 발생하는지 심층 분석한다.

### 핵심 연구 질문

1. **실패 원인 분해**: VLM이 특정 지역에서 실패하는 이유는 무엇인가?
   - OCR/텍스트 인식 문제?
   - 추론 능력 부족?
   - 지식 결핍?

2. **지역별 병목**: 각 지역에서 주요 실패 유형이 다른가?

3. **모델별 차이**: 왜 Gemini는 일본어에서 강하고 GPT는 약한가?

---

## 6.2 Failure Taxonomy

### 중요 발견: 데이터셋 라벨링 오류

Human annotation 검증 과정에서 중요한 발견이 있었다. 84개 실패 문항을 전문가가 수동으로 검토한 결과, **46개(54.8%)**가 데이터셋 자체의 라벨링 오류로 판명되었다. 특히 India 문항에서 라벨링 오류율이 **97.1%**에 달했다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    LABELING ERROR ANALYSIS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Total annotated questions:        84                            │
│  Labeling errors (dataset issue):  46 (54.8%)                   │
│  Valid VLM failures:               38 (45.2%)                   │
│                                                                  │
│  Labeling Error Rate by Nation:                                  │
│  ─────────────────────────────────────────────                   │
│  India        34/35  (97.1%)  ████████████████████████████████  │
│  EU            8/19  (42.1%)  ████████████                       │
│  Japan         4/16  (25.0%)  ████████                           │
│  South Korea   1/13  ( 7.7%)  ██                                 │
│  Taiwan        0/2   ( 0.0%)                                     │
│                                                                  │
│  → India 데이터의 심각한 라벨링 문제 확인                        │
│  → 향후 벤치마크 품질 관리의 중요성 시사                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Primary Failure Categories (라벨링 오류 제외, n=38)

라벨링 오류를 제외한 38개 유효 실패 사례를 분석했다.

```
┌─────────────────────────────────────────────────────────────────┐
│                  FAILURE CATEGORY DISTRIBUTION                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Category                          Count    %                    │
│  ─────────────────────────────────────────────────               │
│  시각 실패 (Visual Failure)         29    76.3%  ████████████████│
│  파싱 실패 (Parsing Failure)         4    10.5%  ███             │
│  기타 (Other)                        3     7.9%  ██              │
│  도메인 지식 (Domain Knowledge)       2     5.3%  █              │
│                                                                  │
│  Total valid failures: 38                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 핵심 발견: 인식(Perception) 병목이 지배적

```
┌─────────────────────────────────────────────────────────────────┐
│                    PERCEPTION vs REASONING                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Perception-related failures (Visual + Parsing):                 │
│    29 + 4 = 33 cases (86.8%)                                    │
│                                                                  │
│  Reasoning/Knowledge failures:                                   │
│    2 cases (5.3%)                                               │
│                                                                  │
│  Other:                                                          │
│    3 cases (7.9%)                                               │
│                                                                  │
│  → VLM failures are primarily about "reading", not "thinking"    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**시사점**: Human annotation으로 검증된 유효 실패 사례에서도 VLM이 시험 문제를 틀리는 주요 원인은 "추론을 못해서"가 아니라 "읽기를 못해서"이다 (86.8% vs 5.3%). 이는 기존의 "VLM은 추론이 약하다"는 통념과 다른 발견이다.

---

## 6.3 Regional Bottleneck Analysis

### 지역별 유효 실패 분포 (라벨링 오류 제외)

```
Table: Valid Failure Distribution by Region (Human Verified, n=38)
──────────────────────────────────────────────────────────────────
Region       | Valid Fails | Labeling Errors | Total | Valid %
──────────────────────────────────────────────────────────────────
Japan        |     12      |        4        |   16  |  75.0%
South Korea  |     12      |        1        |   13  |  92.3%
EU           |     11      |        8        |   19  |  57.9%
Taiwan       |      2      |        0        |    2  | 100.0%
India        |      1      |       34        |   35  |   2.9%
──────────────────────────────────────────────────────────────────
Note: India의 대부분 실패(97.1%)가 데이터셋 라벨링 오류로 판명됨
```

### 지역별 실패 유형 (유효 실패 기준, n=38)

```
Table: Primary Failure Categories by Region (Valid Failures Only)
──────────────────────────────────────────────────────────────────
Region       | Visual | Parsing | Domain Knowledge | Other
──────────────────────────────────────────────────────────────────
Japan (n=12) |   10   |    7    |        2         |   2
S.Korea (n=12)|   9   |   10    |        1         |   1
EU (n=11)    |    7   |    4    |        1         |   0
Taiwan (n=2) |    2   |    0    |        0         |   0
India (n=1)  |    1   |    1    |        0         |   0
──────────────────────────────────────────────────────────────────
Note: 각 문제는 여러 실패 유형을 동시에 가질 수 있음
```

### 지역별 병목 유형 요약

```
┌─────────────────────────────────────────────────────────────────┐
│                 REGIONAL BOTTLENECK PATTERNS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  JAPAN (12 valid failures)                                      │
│  ├─ Primary bottleneck: Visual failure (83.3%)                  │
│  ├─ Secondary: Parsing failure (58.3%)                          │
│  ├─ Perception-related: ~83%                                    │
│  └─ Recommendation: Japanese OCR + layout recognition 강화      │
│                                                                  │
│  SOUTH KOREA (12 valid failures)                                │
│  ├─ Primary bottleneck: Parsing failure (83.3%)                 │
│  ├─ Secondary: Visual failure (75.0%)                           │
│  ├─ Perception-related: ~92%                                    │
│  └─ Recommendation: Hangul OCR + 표 구조 파악 강화              │
│                                                                  │
│  EU (11 valid failures)                                         │
│  ├─ Primary bottleneck: Visual failure (63.6%)                  │
│  ├─ Secondary: Parsing failure (36.4%)                          │
│  ├─ Perception-related: ~91%                                    │
│  └─ Note: Latin 문자에서도 시각 실패 발생                       │
│                                                                  │
│  TAIWAN (2 valid failures)                                      │
│  ├─ Primary bottleneck: Visual failure (100%)                   │
│  ├─ Perception-related: 100%                                    │
│  └─ Note: 샘플 수 적어 통계적 해석 제한적                        │
│                                                                  │
│  INDIA (1 valid failure - 대부분 라벨링 오류)                   │
│  ├─ 34/35 문항이 라벨링 오류로 판명                             │
│  ├─ VLM 실패가 아닌 데이터셋 품질 문제                          │
│  └─ Recommendation: 데이터셋 재검증 필요                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 핵심 통찰

1. **아시아 3국 (Japan, Korea, India)**: 인식(Perception) 병목이 지배적
   - 문자 체계 복잡성이 성능 저하의 주원인
   - OCR/텍스트 인식 개선 시 큰 성능 향상 예상

2. **EU**: 추론(Reasoning) 병목이 지배적
   - 라틴 문자는 잘 읽음 → OCR 문제 없음
   - 실패는 "지식/추론 부족" 때문
   - 이것이 "정상적인" 실패 패턴

3. **Taiwan**: 전반적으로 실패가 적음
   - 번체 중국어가 예상보다 잘 처리됨
   - 일부 OCR 실패만 존재

---

## 6.4 Model-Specific Insights

### Gemini vs GPT: 아시아 언어 처리 능력 격차

```
┌─────────────────────────────────────────────────────────────────┐
│                  GEMINI vs GPT COMPARISON                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Region      Gemini-2.5-pro   GPT-4o      Gap                   │
│  ─────────────────────────────────────────────                   │
│  Japan          87.6%         26.0%      +61.6%p  ← Massive!    │
│  Korea          91.1%         33.2%      +57.9%p  ← Massive!    │
│  India          69.2%         41.0%      +28.2%p                │
│  Taiwan         95.5%         66.7%      +28.8%p                │
│  EU             88.1%         63.7%      +24.4%p                │
│                                                                  │
│  Pattern:                                                        │
│  • Gemini: Consistent across all regions (69-95%)               │
│  • GPT-4o: Huge variance (26-67%), collapses on Asian scripts   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 원인 분석

**Gemini 계열의 아시아 언어 강점:**
- Google의 다국어 학습 데이터 (YouTube, Google Search, Google Translate)
- 내장 OCR 품질이 높음 (Google Lens 기술 활용 추정)
- 멀티모달 인코더가 다양한 문자체계에 최적화

**GPT 계열의 아시아 언어 약점:**
- 학습 데이터가 영어 중심
- 복잡한 문자체계(한자, 한글)에서 OCR 품질 저하
- EU 문서에서는 준수하지만 (63.7%), 일본 문서에서 급락 (26.0%)

### 모델별 지역 분산 분석

```
Models with HIGHEST regional variance (most unstable):
──────────────────────────────────────────────────────
Model                    Nation Variance   Interpretation
──────────────────────────────────────────────────────
Gemini-2.5-flash-lite       702.09        Extreme collapse on Asian
GPT-4o                      266.00        Large regional gaps
Gemini-2.5-flash            217.45        Moderate instability
Claude-Sonnet-4             197.74        Moderate instability

Models with LOWEST regional variance (most stable):
──────────────────────────────────────────────────────
Model                    Nation Variance   Interpretation
──────────────────────────────────────────────────────
llava-1.5-7b                  0.37        Uniformly poor everywhere
LLaVA-NeXT-Video-7B           0.37        Uniformly poor everywhere
llava-1.5-13b                 2.93        Consistently mediocre
Phi-3.5-vision                4.15        Consistently poor
```

**관찰**: 
- 최고 성능 모델(Gemini-2.5-pro)은 분산이 중간 수준 (80.85)
- 최악의 분산을 보이는 모델은 특정 지역에서 극단적으로 실패
- 일관되게 낮은 분산을 보이는 모델은 "모든 곳에서 똑같이 나쁨"

---

## 6.5 Script Complexity Analysis

### 문자 체계별 난이도

```
┌─────────────────────────────────────────────────────────────────┐
│                  SCRIPT COMPLEXITY HIERARCHY                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Most Complex                                    Least Complex   │
│  ◀────────────────────────────────────────────────────────────▶ │
│                                                                  │
│  Japanese          Korean         Devanagari     Chinese    Latin│
│  (Kanji+Hira+Kata) (Hangul)      (Hindi)        (Trad.)   (EN)  │
│     32.5%           38.6%          32.6%         54.9%    49.9% │
│                                                                  │
│  Characteristics:                                                │
│  • Japanese: 3 scripts mixed, complex layout, vertical text      │
│  • Korean: Syllabic blocks, dense information per character      │
│  • Devanagari: Complex ligatures, diacritics                     │
│  • Chinese: Ideographic but single script system                 │
│  • Latin: Familiar to most models, simple structure              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 왜 Japanese가 가장 어려운가?

1. **세 가지 문자 체계 혼용**: 漢字(한자) + ひらがな(히라가나) + カタカナ(가타카나)
2. **세로쓰기(縦書き)**: 일부 문서에서 세로 방향 텍스트
3. **한자 복잡성**: 수천 개의 한자, 유사한 형태의 문자 다수
4. **밀집된 정보**: 한 문자에 많은 정보 함축

### 왜 Taiwan이 Japan보다 쉬운가?

- **단일 문자 체계**: 번체 중국어만 사용
- **가로쓰기 위주**: 세로쓰기 거의 없음
- **문맥 일관성**: 한자만 사용하므로 패턴 학습이 용이

---

## 6.6 Failure Case Examples

### Case 1: Japanese Law Question (GPT-4o 실패)

```
┌─────────────────────────────────────────────────────────────────┐
│  Question: 日本国憲法における基本的人権の制限に関する記述...      │
│                                                                  │
│  GPT-4o Response:                                                │
│  - Misread "憲法" (constitution) as "意法"                       │
│  - Incorrectly parsed legal terminology                          │
│  - Selected wrong answer due to OCR error                        │
│                                                                  │
│  Gemini-2.5-pro Response:                                        │
│  - Correctly read all kanji characters                           │
│  - Understood legal context                                      │
│  - Selected correct answer                                       │
│                                                                  │
│  Failure Type: vertical_nonlatin_script + ocr_text_recognition  │
└─────────────────────────────────────────────────────────────────┘
```

### Case 2: Korean Administrative Question (Multiple models 실패)

```
┌─────────────────────────────────────────────────────────────────┐
│  Question: 다음 표를 보고 지방자치단체의 재정자립도를 계산하시오. │
│                                                                  │
│  [Complex table with Korean text and numbers]                    │
│                                                                  │
│  Common Failure Pattern:                                         │
│  - Table structure not correctly parsed                          │
│  - Korean column headers misread                                 │
│  - Numerical calculations correct, but wrong cells selected      │
│                                                                  │
│  Failure Type: table_structure + vertical_nonlatin_script       │
└─────────────────────────────────────────────────────────────────┘
```

### Case 3: EU Reasoning Question (Multiple models 실패)

```
┌─────────────────────────────────────────────────────────────────┐
│  Question: Based on the EU regulation excerpt, determine which   │
│  of the following scenarios constitutes a violation...           │
│                                                                  │
│  Common Failure Pattern:                                         │
│  - Text correctly read (Latin alphabet)                          │
│  - Legal reasoning chain incomplete                              │
│  - Failed to apply regulation to specific scenario               │
│                                                                  │
│  Failure Type: pure_reasoning_knowledge                         │
│                                                                  │
│  → This is the "normal" failure mode we expect from VLMs        │
│  → Asian failures are "abnormal" (should be fixable with OCR)   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6.7 Visual Element Impact

### 시각적 요소별 영향

```
Visual Elements in Failure Cases (N=176):
──────────────────────────────────────────
Element        | Count | % of failures
──────────────────────────────────────────
Has Math       |  68   |    38.6%
Has Table      |  26   |    14.8%
Has Diagram    |  24   |    13.6%
Has Handwriting|   0   |     0.0%
──────────────────────────────────────────
```

### 수학 기호 문제의 심각성

- 전체 실패의 38.6%에 수학 기호 포함
- 특히 **일본 + 수학** 조합이 치명적 (물리, 화학, 수학 과목)
- 수식 내 변수와 일본어 텍스트 혼용 시 OCR 품질 급락

---

## 6.8 Summary: Diagnostic Insights

```
┌─────────────────────────────────────────────────────────────────┐
│                    DIAGNOSTIC CONCLUSIONS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PERCEPTION DOMINATES REASONING                               │
│     • 72% of failures are perception-related (OCR, script)       │
│     • Only 11% are pure reasoning/knowledge failures             │
│     • VLMs "can't read", not "can't think"                      │
│                                                                  │
│  2. REGIONAL BOTTLENECKS ARE DISTINCT                           │
│     • Asia (JP, KR, IN): Perception bottleneck                  │
│     • EU: Reasoning bottleneck (normal failure mode)            │
│     • Taiwan: Minimal failures (best handled)                    │
│                                                                  │
│  3. MODEL-SPECIFIC GAPS ARE HUGE                                │
│     • Gemini: Strong Asian language support (87% Japan)          │
│     • GPT: Weak Asian language support (26% Japan)               │
│     • Gap of 61.6%p on SAME exam!                               │
│                                                                  │
│  4. SCRIPT COMPLEXITY MATTERS MOST                              │
│     • Japanese (3 scripts) > Korean > Devanagari > Chinese > Latin│
│     • Vertical text adds additional difficulty                   │
│     • Math + Non-Latin = Worst combination                       │
│                                                                  │
│  IMPLICATION FOR GOVERNMENT AI:                                  │
│  • Asian countries: Prioritize OCR quality over reasoning        │
│  • EU countries: Focus on domain knowledge enhancement           │
│  • All: Model selection matters (3x performance gap possible)    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Figure/Table 목록 (이 섹션)

| # | Type | Content | Priority |
|---|------|---------|----------|
| Table 3 | Distribution | Failure category distribution | HIGH |
| Table 4 | Breakdown | Regional bottleneck patterns | HIGH |
| Figure 5 | Bar chart | Perception vs Reasoning failures | HIGH |
| Figure 6 | Comparison | Gemini vs GPT regional performance | MEDIUM |
| Figure 7 | Examples | Failure case images (3 examples) | MEDIUM |
