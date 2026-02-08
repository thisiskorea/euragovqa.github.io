# Abstract (한국어 초안)

---

## Final Version (250 words)

### 한국어

Vision-Language Model(VLM)의 성능이 급격히 향상되면서, 각국 정부는 행정 문서 처리와 민원 자동화에 AI 도입을 적극 검토하고 있다. 그러나 정부 시스템에 VLM을 배포하기 전에, 핵심 질문에 답해야 한다: **"영어 벤치마크에서 우수한 모델이 다른 나라 공문서에서도 신뢰할 수 있는가?"**

이 질문에 답하기 위해, 본 연구는 **EuraGovExam**을 제안한다—한국, 일본, 대만, 인도, EU 5개 지역의 실제 공무원 시험 8,000+개로 구성된 다국어 멀티모달 벤치마크이다. 공무원 시험은 관할권별 특수 지식을 실제 스캔 문서 형태로 평가하므로, 정부 AI 신뢰성 검증에 이상적인 테스트베드이다.

23개 최신 VLM을 평가한 결과, 우려스러운 발견을 했다: **지역/언어에 따른 성능 격차가 과목 난이도 격차보다 3.9배 더 크다** (η²=0.126 vs 0.043, p<0.01). 구체적으로, GPT-4o는 EU 문서에서 63.7%를 달성하지만 일본 문서에서는 26.0%로 추락한다—거의 랜덤 수준이다. 이러한 패턴은 23개 모델 중 82.6%에서 일관되게 관찰된다.

이 결과는 VLM의 "범용 지능"에 대한 과신을 경고한다. 정부가 AI를 도입하기 전에 대상 지역/언어에 특화된 검증이 필수이며, EuraGovExam은 이러한 검증을 위한 도구를 제공한다.

---

### English Version

With Vision-Language Models (VLMs) achieving expert-level performance on professional examinations, governments worldwide are actively exploring AI deployment for administrative document processing and citizen services. However, before deploying VLMs in high-stakes government systems, we must answer a critical question: **"Can a model that excels on English benchmarks reliably process documents in other languages?"**

To address this question, we introduce **EuraGovExam**, a multilingual multimodal benchmark comprising 8,000+ authentic civil service examination questions from five Eurasian regions: Korea, Japan, Taiwan, India, and the EU. Civil service exams test jurisdiction-specific knowledge embedded in real scanned documents, making them an ideal testbed for validating government AI readiness.

Evaluating 23 state-of-the-art VLMs, we uncover an alarming finding: **regional/linguistic effects dominate task/subject effects by a 3.9× variance ratio** (η²=0.126 vs 0.043, p<0.01). Specifically, GPT-4o achieves 63.7% accuracy on EU documents but plummets to 26.0% on Japanese documents—barely above random chance. This pattern is consistent across 82.6% of evaluated models.

These results challenge the assumption of "universal AI capability" and demonstrate that VLM reliability varies dramatically by region. Region-specific validation is not optional but essential before government deployment. EuraGovExam provides the benchmark and diagnostic tools needed for such validation.

---

## Key Elements Checklist

| Element | Present? | Content |
|---------|----------|---------|
| Context/Background | ✓ | VLM performance improving, governments adopting |
| Research Question | ✓ | Can we trust VLMs across languages? |
| Dataset Introduction | ✓ | EuraGovExam: 8,000+, 5 regions, civil service |
| Key Finding | ✓ | 3.9× variance ratio, regional > task |
| Specific Example | ✓ | GPT-4o: EU 63.7% vs Japan 26.0% |
| Consistency Evidence | ✓ | 82.6% of models show same pattern |
| Implication | ✓ | Region-specific validation required |
| Dataset Utility | ✓ | Provides tools for validation |

---

## Word Count Check

- Target: 250 words
- Current English version: ~240 words ✓

---

## Alternative Versions

### Version A: More Dramatic (Hook-focused)

> GPT-4o passes the US bar exam but scores only 26% on Japanese civil service exams—barely above random chance. This is not an anomaly: we show that VLM performance varies by **3.9×** depending on the target language, a finding with serious implications for governments planning AI deployment.
>
> We introduce EuraGovExam, a benchmark of 8,000+ authentic civil service examinations from Korea, Japan, Taiwan, India, and the EU. Unlike existing benchmarks using synthetically rendered images and universal knowledge, civil service exams test jurisdiction-specific expertise embedded in real scanned documents.
>
> [Continue with findings and implications...]

### Version B: Dataset-First (Traditional)

> We present EuraGovExam, a multilingual multimodal benchmark for evaluating Vision-Language Models on real-world government document processing. The dataset comprises 8,000+ civil service examination questions from five Eurasian regions...
>
> [Continue with methodology and findings...]

---

## Recommendation

**Use the "Final Version" above.** It balances:
1. Practical motivation (government AI adoption)
2. Clear research question
3. Strong quantitative findings
4. Direct implications

The dramatic hook version (A) might be too aggressive for NeurIPS. The dataset-first version (B) buries the key finding.
