# Section 2: Related Work (한국어 초안)

---

## 2.1 Multimodal Exam and Knowledge Benchmarks

### 주요 벤치마크 개요

최근 VLM 평가를 위한 다양한 멀티모달 벤치마크가 제안되었다.

**MMMU (Yue et al., 2024)**는 대학 수준의 멀티모달 문제를 포함하며, 과학, 예술, 비즈니스 등 30개 과목을 다룬다. 그러나 영어 중심이며, 이미지가 깨끗하게 렌더링되어 있어 실제 문서의 노이즈와 복잡성을 반영하지 못한다.

**MathVista (Lu et al., 2024)**는 수학적 시각 추론에 특화되어 있으며, 다양한 시각적 맥락(차트, 그래프, 기하학)에서 수학 문제를 평가한다. 단일 도메인에 집중하여 일반적인 문서 이해 능력을 평가하기 어렵다.

**EXAMS-V (Das et al., 2024)**는 본 연구와 가장 관련이 깊다. 16개 언어, 21,000개 학교 시험 문제를 포함하며, 다국어 멀티모달 평가를 시도한다. 그러나 중요한 차이점이 있다:

| Aspect | EXAMS-V | EuraGovExam (Ours) |
|--------|---------|-------------------|
| Source | School curriculum exams | Civil service exams |
| Knowledge | Universal (math, science) | Jurisdiction-specific |
| Images | Rendered from text | Real scanned documents |
| Transferability | Translatable | NOT translatable |
| Ground truth | Crowd-sourced | Official government keys |

### 핵심 차별점

기존 벤치마크들은 **보편적 지식**(수학, 과학)을 평가한다. 이러한 지식은 언어와 무관하게 전이될 수 있다—한국어로 수학 문제를 풀든 영어로 풀든, 문제의 본질은 동일하다.

반면, 공무원 시험은 **관할권별 특수 지식**을 요구한다:
- 한국 행정법 ≠ 일본 행정법 ≠ EU 행정법
- 번역만으로는 해결되지 않음
- 해당 지역의 법률, 제도, 문화적 맥락에 대한 이해 필요

이 차이가 EuraGovExam의 핵심 기여이다: **VLM이 특정 지역의 문서를 "진정으로" 이해하는지 평가**할 수 있다.

---

## 2.2 Document Understanding Benchmarks

### 문서 이해 벤치마크

**DocVQA (Mathew et al., 2021)**는 산업 문서에 대한 질의응답을 평가한다. 다양한 레이아웃(양식, 표, 영수증)을 포함하지만, 주로 영어 문서에 한정된다.

**ChartQA (Masry et al., 2022)**는 차트와 그래프 이해에 특화되어 있다. 시각적 데이터 해석 능력을 평가하지만, 텍스트 중심 문서 이해와는 다른 영역이다.

**OCRBench (Liu et al., 2024)**는 VLM의 OCR 능력을 체계적으로 평가한다. 텍스트 인식, 공간 이해, 수식 인식 등을 다루며, 본 연구의 진단적 접근과 관련이 있다.

**TextVQA (Singh et al., 2019)**는 자연 이미지 내 텍스트 읽기를 평가한다. 간판, 책 표지 등 짧은 텍스트에 초점을 맞추며, 긴 문서 이해와는 다르다.

### 본 연구와의 관계

기존 문서 이해 벤치마크들은:
1. 영어 중심 (다국어 지원 제한적)
2. 단일 유형에 특화 (차트만, 양식만 등)
3. 지식 추론보다는 정보 추출에 초점

EuraGovExam은:
1. 5개 지역, 4개 문자체계
2. 다양한 문서 유형 (텍스트, 표, 다이어그램 혼합)
3. OCR + 이해 + 추론을 모두 요구

---

## 2.3 Multilingual VLM Evaluation

### 다국어 VLM 연구

**xGQA, MaXM** 등은 다국어 시각 질의응답을 평가한다. 그러나 대부분 영어 데이터셋을 번역하여 생성했으며, 언어별 고유한 특성을 반영하지 못한다.

**Multilingual MMLU variants**는 MMLU를 여러 언어로 번역했다. 번역 품질 이슈가 있으며, 문화적/법적 맥락이 다른 문제를 평가하지 못한다.

### 번역 기반 접근의 한계

```
Example: Legal Question

Original (English): 
"Under which US constitutional amendment is double jeopardy prohibited?"

Translated to Korean:
"미국 헌법 어느 조항에서 이중 위험을 금지하는가?"

Problem:
- 한국 법체계와 무관
- 한국 공무원이 알 필요 없는 지식
- VLM의 "한국 문서 이해 능력"을 평가하지 못함
```

EuraGovExam은 **각 지역의 원본 시험**을 사용하므로, 이러한 한계를 극복한다.

---

## 2.4 VLM Capability Decomposition and Diagnostics

### 기존 진단적 연구

**Perception Bottleneck Studies**: 여러 연구가 VLM의 시각적 인식과 추론 능력을 분리하려 시도했다. Liu et al. (2023)은 이미지 캡셔닝을 통한 성능 분해를, Yang et al. (2024)은 OCR 품질의 영향을 분석했다.

**LogicOCR and Text-centric VLM Analysis**: OCR 능력이 VLM 성능의 주요 병목임을 보인 연구들이 있다. 우리의 4-Track 프로토콜은 이러한 선행 연구를 발전시켜, 체계적인 병목 분해를 제공한다.

### 본 연구의 포지셔닝

우리는 **새로운 방법론**을 제안하는 것이 아니라, 기존 진단적 접근을 **체계적으로 적용**하여 **새로운 발견**을 도출한다:

| Prior Work | Our Work |
|------------|----------|
| Proposed diagnostic methods | Applied methods systematically |
| Tested on limited data | Tested on 8,000+ real exams |
| English/single-language | 5 regions, 4 writing systems |
| Method as contribution | Finding as contribution |

**핵심 발견**: 지역 효과가 과목 효과를 3.9배 압도한다는 것은 기존 연구에서 보고되지 않은 새로운 발견이다.

---

## 2.5 Government AI Adoption and Trust

### 정부 AI 도입 연구

정부의 AI 도입에 대한 연구는 주로 정책적, 윤리적 관점에서 이루어졌다:
- AI 거버넌스 프레임워크 (EU AI Act, US OMB Memo)
- 공정성과 편향 문제
- 설명 가능성 요구사항

그러나 **기술적 신뢰성**—특히 다국어 환경에서의 성능 일관성—에 대한 연구는 부족하다.

### 본 연구의 위치

EuraGovExam은 정부 AI 도입의 **기술적 전제조건**을 검증한다:
- 정책/윤리 논의 이전에, 모델이 기본적으로 작동하는지 확인
- "이 AI를 도입해도 되는가?" → "이 AI가 우리 문서를 읽을 수 있는가?"

---

## Key Citations (영어 작성 시 참조)

```bibtex
@article{yue2024mmmu,
  title={MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark},
  author={Yue, Xiang and others},
  journal={CVPR},
  year={2024}
}

@article{lu2024mathvista,
  title={MathVista: Evaluating Math Reasoning in Visual Contexts},
  author={Lu, Pan and others},
  journal={ICLR},
  year={2024}
}

@article{das2024examsv,
  title={EXAMS-V: A Multi-Discipline Multilingual Multimodal Exam Benchmark},
  author={Das, Rocktim and others},
  journal={arXiv},
  year={2024}
}

@article{mathew2021docvqa,
  title={DocVQA: A Dataset for VQA on Document Images},
  author={Mathew, Minesh and others},
  journal={WACV},
  year={2021}
}

@article{liu2024ocrbench,
  title={OCRBench: On the Hidden Mystery of OCR in Large Multimodal Models},
  author={Liu, Yuliang and others},
  journal={arXiv},
  year={2024}
}
```

---

## Section Summary

```
Related Work의 핵심 메시지:

1. 기존 벤치마크의 한계
   - 영어 중심
   - 보편적 지식만 평가
   - 깨끗한 렌더링 이미지

2. EuraGovExam의 차별점
   - 다국어 (5개 지역, 4개 문자체계)
   - 관할권별 특수 지식
   - 실제 스캔 문서

3. 방법론적 포지셔닝
   - 새로운 방법론 X
   - 기존 방법론의 체계적 적용 + 새로운 발견 O
```
