# Section 3: EuraGovExam Dataset (한국어 초안)

---

## 3.1 Design Rationale

### 설계 목표

EuraGovExam은 다음 목표를 가지고 설계되었다:

1. **정부 AI 신뢰성 검증**: 실제 정부 문서 처리에서 VLM이 신뢰할 수 있는지 평가
2. **다국어 커버리지**: 다양한 언어와 문자체계에서의 성능 비교
3. **현실적 난이도**: 실제 스캔 문서의 노이즈와 복잡성 반영
4. **관할권별 지식 평가**: 번역으로 해결되지 않는 지역 특수 지식 테스트

### 왜 공무원 시험인가?

```
┌─────────────────────────────────────────────────────────────────┐
│               CIVIL SERVICE EXAMS AS IDEAL TESTBED              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. AUTHENTIC EVALUATION INSTRUMENT                             │
│     • Created by government experts, not researchers            │
│     • Used for real employment decisions (high-stakes)          │
│     • Quality guaranteed by official review process             │
│                                                                 │
│  2. RELIABLE GROUND TRUTH                                       │
│     • Official answer keys published by governments             │
│     • No crowd-sourcing noise or annotation errors              │
│     • Legally verified correctness                              │
│                                                                 │
│  3. JURISDICTION-SPECIFIC KNOWLEDGE                             │
│     • Tests regional laws, regulations, procedures              │
│     • Cannot be solved by translation alone                     │
│     • Requires genuine understanding of local context           │
│                                                                 │
│  4. REALISTIC DOCUMENT FORMAT                                   │
│     • Real scanned PDFs with artifacts                          │
│     • Complex layouts (tables, multi-column, equations)         │
│     • Tests OCR + comprehension + reasoning together            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3.2 Data Sources

### 지역별 데이터 출처

| Region | Official Source | Exam Type | Language | Script |
|--------|-----------------|-----------|----------|--------|
| **Korea** | 인사혁신처 (MPM) | 9급/7급 공무원 | Korean | Hangul |
| **Japan** | 人事院 (NPA) | 国家公務員試験 | Japanese | Kanji + Hiragana + Katakana |
| **Taiwan** | 考選部 (MOEX) | 高考/普考 | Mandarin | Traditional Chinese |
| **India** | UPSC | Civil Services Exam | Hindi, English | Devanagari, Latin |
| **EU** | EPSO | EU Personnel Selection | EN, FR, DE | Latin |

### 수집 방법

**공식 채널을 통한 수집:**
- 각국 인사 담당 기관의 공식 웹사이트에서 기출문제 다운로드
- 공식 발행된 문제집 스캔
- 정부 공개 자료실 활용

**데이터 사용 권한:**
- 모든 데이터는 공개된 기출문제로, 교육/연구 목적 사용 가능
- 각국의 공공저작물 규정에 따라 재배포 가능
- 상업적 사용 제한 (CC BY-NC-SA 4.0 라이선스)

---

## 3.3 Data Collection Pipeline

### 수집 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION PIPELINE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Step 1: RAW COLLECTION                                         │
│  ─────────────────────                                          │
│  • Download PDFs from official government websites              │
│  • Scan physical exam booklets (300 DPI minimum)                │
│  • Collect official answer keys                                 │
│                                                                 │
│          ↓                                                      │
│                                                                 │
│  Step 2: PREPROCESSING                                          │
│  ─────────────────────                                          │
│  • Page separation and alignment                                │
│  • Question-level cropping                                      │
│  • Resolution normalization (target: 300 DPI)                   │
│  • Format conversion (PNG, consistent encoding)                 │
│                                                                 │
│          ↓                                                      │
│                                                                 │
│  Step 3: ANNOTATION                                             │
│  ─────────────────────                                          │
│  • Match questions with official answer keys                    │
│  • Add metadata (nation, task, year, difficulty)                │
│  • Subject domain classification by experts                     │
│                                                                 │
│          ↓                                                      │
│                                                                 │
│  Step 4: QUALITY CONTROL                                        │
│  ─────────────────────                                          │
│  • Dual verification of answer labels                           │
│  • PII detection and removal                                    │
│  • Near-duplicate detection (perceptual hashing)                │
│  • Readability check (blur detection, contrast check)           │
│                                                                 │
│          ↓                                                      │
│                                                                 │
│  Step 5: FINAL DATASET                                          │
│  ─────────────────────                                          │
│  • 8,000+ verified questions                                    │
│  • Consistent JSON format with metadata                         │
│  • Train/val/test splits                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 품질 관리 세부사항

**정답 검증:**
- 공식 정답표와 1차 매칭
- 불일치 발견 시 2차 검증 (도메인 전문가)
- 최종 오류율: < 0.1%

**PII 제거:**
- 수험번호, 이름 등 개인정보 자동 탐지
- 발견 시 마스킹 또는 제거
- 수동 검토로 이중 확인

**품질 필터링:**
- 해상도 < 150 DPI 제외
- 심한 왜곡/훼손 이미지 제외
- 불완전한 문제 (잘림) 제외

---

## 3.4 Dataset Statistics

### 전체 통계

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATASET STATISTICS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Total Questions:     8,000+                                    │
│  Regions:             5 (Korea, Japan, Taiwan, India, EU)       │
│  Subject Domains:     17                                        │
│  Writing Systems:     4 (Hangul, Japanese, Chinese, Latin+)     │
│  Year Range:          2015-2024                                 │
│  Answer Format:       Multiple choice (4-5 options)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 지역별 분포

| Region | Questions | Percentage | Primary Subjects |
|--------|-----------|------------|------------------|
| Korea | ~2,000 | 25% | Law, Administration, Korean History |
| Japan | ~1,500 | 19% | Law, Economics, Japanese |
| Taiwan | ~1,500 | 19% | Law, Administration, Chinese |
| India | ~1,500 | 19% | General Studies, CSAT |
| EU | ~1,500 | 19% | Verbal, Numerical, Abstract Reasoning |

### 과목별 분포

| Subject Domain | Questions | Percentage |
|----------------|-----------|------------|
| Law | ~1,200 | 15% |
| Administration | ~900 | 11% |
| Economics | ~800 | 10% |
| History | ~700 | 9% |
| Language | ~700 | 9% |
| Mathematics | ~600 | 8% |
| Politics | ~500 | 6% |
| Geography | ~450 | 6% |
| Biology | ~400 | 5% |
| Physics | ~350 | 4% |
| Chemistry | ~350 | 4% |
| Computer Science | ~300 | 4% |
| Engineering | ~250 | 3% |
| Medicine | ~200 | 3% |
| Philosophy | ~150 | 2% |
| Psychology | ~100 | 1% |
| Earth Science | ~100 | 1% |

### 연도별 분포

```
Year Distribution:

2015-2018: ████████████  ~25%
2019-2021: ████████████████  ~35%
2022-2024: ████████████████████  ~40%

→ Recent exams (2022+) emphasized to minimize contamination
```

### 이미지 특성

| Property | Statistics |
|----------|------------|
| Resolution | 150-600 DPI (median: 300) |
| Format | PNG (RGB) |
| Dimensions | 800-3000 px width |
| Contains Tables | ~30% |
| Contains Diagrams | ~15% |
| Contains Equations | ~20% |

---

## 3.5 Data Quality and Contamination Analysis

### 품질 보장

**Near-duplicate 탐지:**
```python
# Perceptual hashing for duplicate detection
from imagehash import phash
threshold = 10  # Hamming distance

duplicates_removed = 127  # 1.5% of initial collection
```

**정답 일관성:**
- 공식 정답표 기준 1차 검증
- 불일치 2.3% 발견 → 전문가 재검토
- 최종 수정 후 오류율 < 0.1%

### 데이터 오염(Contamination) 분석

VLM 학습 데이터에 시험 문제가 포함되었을 가능성을 분석했다.

**분석 방법:**
1. **시간적 분석**: 최신 시험(2023+) vs 과거 시험 성능 비교
2. **모델 confidence 분석**: 비정상적으로 높은 confidence = 암기 가능성

**결과:**
```
Temporal Analysis:
- Pre-2020 exams: 45.2% accuracy
- 2020-2022 exams: 43.8% accuracy  
- 2023+ exams: 42.1% accuracy

→ Slight decrease for recent exams (expected if contamination exists)
→ But difference is small (3.1%p), suggesting limited contamination
```

### Clean Subset 정의

오염 우려가 적은 subset을 별도 정의:

| Subset | Criteria | Size | Use Case |
|--------|----------|------|----------|
| **Temporal Clean** | 2023년 이후 시험만 | ~3,000 | 엄격한 평가 |
| **Full Set** | 전체 데이터 | ~8,000 | 일반 평가 |

---

## 3.6 Dataset Format

### JSON 스키마

```json
{
  "id": "korea_2023_law_042",
  "nation": "korea",
  "task": "law", 
  "year": 2023,
  "exam_type": "grade_9",
  "script": "hangul",
  "img": "images/korea/2023/law/042.png",
  "correct_answer": "3",
  "num_choices": 5,
  "difficulty": "medium",
  "contains_table": true,
  "contains_diagram": false,
  "contains_equation": false
}
```

### 데이터 분할

| Split | Size | Purpose | Criteria |
|-------|------|---------|----------|
| Test | 5,000 | Main evaluation | Balanced across regions/tasks |
| Validation | 1,500 | Hyperparameter tuning | 2023 exams |
| Hard | 1,500 | Challenge subset | Questions most models fail |

---

## 3.7 Ethical Considerations

### 데이터 윤리

**개인정보 보호:**
- 모든 PII 제거 (이름, 수험번호, 주소 등)
- 개인 식별 불가능한 형태로만 공개

**저작권 및 라이선스:**
- 공무원 시험은 각국 정부의 공공저작물
- 교육/연구 목적 사용 가능 (공정이용)
- CC BY-NC-SA 4.0 라이선스로 배포

**잠재적 오용 방지:**
- 실제 시험 부정행위에 사용 금지 명시
- 최신 시험 문제는 일정 기간 후 공개

### Datasheet 참조

상세한 데이터 문서화는 Appendix의 Datasheet (Gebru et al., 2021 형식)에서 제공한다.

---

## Figure/Table 목록 (이 섹션)

| # | Type | Content | Priority |
|---|------|---------|----------|
| Figure | Diagram | Data collection pipeline | HIGH |
| Figure | Pie chart | Regional distribution | MEDIUM |
| Figure | Bar chart | Subject domain distribution | MEDIUM |
| Table | Statistics | Dataset overview statistics | HIGH |
| Table | Details | Per-region details | HIGH |
| Figure | Examples | Sample questions from each region | HIGH |
