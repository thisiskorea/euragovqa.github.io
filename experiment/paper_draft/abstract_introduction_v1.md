# EuraGovExam: Diagnosing Regional and Task Bottlenecks in Multilingual Document VLMs

## Abstract

**[Problem]** Vision-Language Models (VLMs) are increasingly deployed for document understanding, yet their performance on real-world multilingual documents—particularly those with complex layouts and diverse scripts—remains understudied. Existing benchmarks often use synthetically rendered or text-first inputs, obscuring the interaction between visual perception and linguistic reasoning.

**[Dataset]** We introduce **EuraGovExam**, a benchmark of 8,000+ authentic civil service examination questions from five Eurasian regions (Korea, Japan, Taiwan, India, EU), spanning four writing systems and 17 subject domains. Unlike academic exams, civil service tests require jurisdiction-specific knowledge embedded in document-like scanned images.

**[Method & Finding]** Through systematic evaluation of 23 VLMs using a diagnostic protocol, we uncover a striking finding: **regional/jurisdictional effects dominate task/subject effects by a 3.9× variance ratio** (η²=0.126 vs η²=0.043, p<0.01). This pattern holds across 82.6% of evaluated models. Japan emerges as the hardest region (32.5% accuracy) while Taiwan is easiest (54.9%), revealing a 22.4 percentage-point performance gap unexplained by task difficulty alone.

**[Implication]** These results suggest that VLM failures on multilingual documents stem more from region-specific bottlenecks (script complexity, OCR quality, jurisdictional knowledge) than from reasoning difficulty per se. EuraGovExam provides actionable diagnostics for targeted VLM improvements.

---

## 1. Introduction

### 1.1 Motivation: The Gap Between Lab and Real-World Documents

Vision-Language Models have achieved impressive performance on standard multimodal benchmarks [citations: MMMU, MathVista, etc.]. However, real-world document understanding presents unique challenges that current benchmarks fail to capture:

1. **Document-like images**: Real documents contain noise, scanning artifacts, complex layouts, and embedded visual elements (tables, diagrams, stamps) that differ fundamentally from clean renders or web-scraped images.

2. **Script diversity**: Beyond Latin scripts, practical deployment requires handling logographic (Chinese, Japanese), syllabic (Korean Hangul), and abugida (Devanagari) writing systems—each with distinct OCR challenges.

3. **Jurisdiction-specific knowledge**: Unlike factual knowledge that transfers across languages, civil service exams test region-specific legal frameworks, administrative procedures, and cultural contexts.

### 1.2 Why Civil Service Exams?

We focus on civil service examinations for three key reasons:

| Property | Academic Exams (EXAMS-V) | Civil Service Exams (Ours) |
|----------|--------------------------|----------------------------|
| Knowledge type | Universal (math, science) | Jurisdiction-specific |
| Image source | Rendered from text | Scanned real documents |
| Annotation quality | Crowd-sourced | Official answer keys |
| Stakes | Educational | Employment (high-stakes) |

Civil service exams are **naturally occurring** evaluation instruments with official ground truth, testing the intersection of visual perception, language understanding, and region-specific knowledge that no existing benchmark captures.

### 1.3 Key Finding: Region Dominates Task

Our central empirical finding challenges the conventional assumption that task difficulty (mathematics vs. law vs. biology) is the primary driver of VLM performance variance on exam benchmarks.

**Through rigorous statistical analysis, we find:**

```
┌─────────────────────────────────────────────────────────────────┐
│  MAIN RESULT: Regional Effect >> Task Effect                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Variance Ratio:     Nation / Task = 3.9×                       │
│                                                                 │
│  Effect Sizes:       η²(Nation) = 0.126 (medium)                │
│                      η²(Task)   = 0.043 (small)                 │
│                                                                 │
│  Significance:       F(4,110) = 3.95, p = 0.005**               │
│                      (Task effect not significant, p = 0.40)    │
│                                                                 │
│  Consistency:        Nation dominates in 82.6% of models        │
│                      (Binomial test p = 0.0013)                 │
│                                                                 │
│  Performance Gap:    Japan 32.5% → Taiwan 54.9% (Δ = 22.4pp)    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

This finding has important implications:
- **For VLM developers**: Improving performance requires addressing region-specific bottlenecks (OCR for complex scripts, jurisdictional knowledge), not just general reasoning.
- **For benchmark designers**: Controlling for regional confounds is essential when comparing models across multilingual datasets.
- **For practitioners**: Deployment readiness varies dramatically by target region; a model "good at exams" may fail catastrophically in specific locales.

### 1.4 Contributions

We make the following contributions:

1. **Diagnostic Protocol (Primary Contribution)**: A systematic evaluation framework that decomposes VLM performance into perception, language, and jurisdictional knowledge components, enabling targeted bottleneck identification.

2. **Key Empirical Finding**: Regional/jurisdictional effects dominate task effects (3.9× variance ratio, η² = 0.126 vs 0.043), validated across 82.6% of 23 evaluated models—a finding with direct implications for VLM development priorities.

3. **EuraGovExam Dataset**: 8,000+ document-like civil-service exam questions from 5 Eurasian regions across 4 writing systems, with official answer keys and comprehensive metadata.

4. **Actionable Insights**: Region-specific bottleneck patterns (e.g., Japan's OCR challenge vs. India's reasoning challenge) that guide targeted VLM improvements.

### 1.5 Paper Organization

- **Section 2**: Related work on multimodal benchmarks and diagnostic evaluation
- **Section 3**: EuraGovExam dataset construction and statistics
- **Section 4**: Evaluation protocol and experimental setup
- **Section 5**: Main results: Regional vs. task effects
- **Section 6**: Diagnostic analysis: Bottleneck decomposition by region
- **Section 7**: Ablation studies and robustness analysis
- **Section 8**: Discussion and implications for VLM development
- **Section 9**: Limitations and future work

---

## Alternative Abstract Versions

### Version A: Findings-First (Recommended for NeurIPS D&B)

> Civil-service examinations test jurisdiction-specific knowledge through document-like images—a capability gap unexplored by existing VLM benchmarks. We introduce EuraGovExam, 8,000+ authentic exam questions from five Eurasian regions, and conduct a systematic diagnostic study of 23 VLMs. Our key finding challenges conventional assumptions: **regional effects dominate task effects by 3.9×** (η²=0.126 vs 0.043, p<0.01). This pattern—consistent across 82.6% of models—reveals that VLM failures stem more from region-specific bottlenecks (script complexity, jurisdictional knowledge) than from reasoning difficulty. Japan shows a 22.4 percentage-point gap from Taiwan, unexplained by task composition. We provide a diagnostic protocol that decomposes these bottlenecks into actionable improvement targets.

### Version B: Dataset-First

> We present EuraGovExam, a multilingual multimodal benchmark of 8,000+ civil service examination questions from Korea, Japan, Taiwan, India, and EU. Unlike academic exams testing universal knowledge, civil service tests require jurisdiction-specific expertise embedded in scanned document images. Through evaluation of 23 state-of-the-art VLMs, we find that regional factors explain 3.9× more performance variance than task factors (η²=0.126 vs 0.043). This finding, consistent across 82.6% of models, suggests that targeted improvements to region-specific bottlenecks—rather than general reasoning enhancements—are needed for practical multilingual deployment.

---

## Key Statistics for Paper (Ready to Use)

| Statistic | Value | Context |
|-----------|-------|---------|
| Variance ratio (Nation/Task) | 3.9× | Main finding |
| Nation effect size (η²) | 0.126 | Medium effect |
| Task effect size (η²) | 0.043 | Small effect |
| Nation F-statistic | F(4,110) = 3.95 | p = 0.005** |
| Task F-statistic | F(16,368) = 1.05 | p = 0.40 (n.s.) |
| Models where nation dominates | 82.6% (19/23) | Consistency |
| Binomial test p-value | 0.0013 | Significant |
| Hardest region | Japan (32.5%) | - |
| Easiest region | Taiwan (54.9%) | - |
| Performance gap | 22.4 pp | Japan → Taiwan |
| Closed models ratio | 2.45× | Nation/Task |
| Open models ratio | 1.85× | Nation/Task |

---

## Framing Comparison: Old vs New

| Aspect | OLD (AAAI Rejected) | NEW (NeurIPS D&B) |
|--------|---------------------|-------------------|
| Main claim | "Novel 3-Track Protocol" | "Regional effects dominate task effects" |
| Novelty locus | Methodology | Empirical finding |
| Dataset role | Primary contribution | Supporting contribution |
| Protocol role | Primary contribution | Diagnostic tool |
| Key number | "8,000+ questions" | "3.9× variance ratio" |
| Reviewer takeaway | "Another benchmark" | "Surprising finding with implications" |

---

## Next Steps

1. [ ] Finalize abstract choice (Version A recommended)
2. [ ] Write Related Work section (position against EXAMS-V, MMMU, MathVista)
3. [ ] Create Introduction figure (variance decomposition visualization)
4. [ ] Draft Croissant metadata for NeurIPS D&B requirements
