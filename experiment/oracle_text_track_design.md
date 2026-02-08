# Oracle Text Track (B2) Design Document

## Problem Statement
The current 3-Track Protocol cannot isolate OCR quality from text comprehension:
- Track A: Image → VLM (tests vision + reasoning)
- Track B: OCR Text → LLM (tests OCR quality + reasoning)
- Track C: Image + OCR Text → VLM (tests multimodal fusion)

**Issue**: Δ(B-A) conflates two variables:
1. OCR extraction quality
2. Modality difference (vision vs text)

## Proposed Solution: 4-Track Protocol with Oracle Text

### Track B2: Oracle Text Track
- **Input**: Human-transcribed "ground truth" text → LLM
- **Purpose**: Isolate OCR quality from reasoning ability

### New Delta Analysis
| Delta | Meaning | Causal Interpretation |
|-------|---------|----------------------|
| Δ(B1-A) | OCR Text vs Image | Mixed (OCR + modality) |
| Δ(B2-A) | Perfect Text vs Image | Pure modality gap |
| Δ(B2-B1) | Perfect Text vs OCR Text | **Pure OCR quality gap** |
| Δ(C-B2) | Multimodal vs Perfect Text | Image contribution |

## Implementation Challenges

### Challenge 1: No Human Transcription in Dataset
The EuraGovExam dataset only contains:
- `nation`: Country of origin
- `correct answer`: Ground truth answer (A-E)
- `img`: Exam question image
- `task`: Subject category

**No `text` or `transcription` field exists.**

### Challenge 2: Scale of Manual Transcription
- 8,000 questions across 5 regions
- Multiple languages (Korean, Japanese, Chinese, German, French, Hindi, etc.)
- Estimated effort: 200+ hours of manual work

## Feasible Alternatives

### Alternative 1: Subset Human Annotation (RECOMMENDED)
- Select 200 stratified samples (40 per region)
- Manually transcribe these 200 questions
- Run B2 track only on this subset
- **Effort**: ~20 hours
- **Benefit**: Direct causal measurement of OCR gap

### Alternative 2: High-Quality OCR Baseline
- Use state-of-the-art OCR (GPT-4V OCR mode, Claude Vision)
- Compare against Gemini OCR
- **Limitation**: Still not "perfect" oracle text

### Alternative 3: Synthetic Oracle via Question Reconstruction
- Use GPT-4 to "reconstruct" question text from correct answers + context
- **Limitation**: Introduces new confounders

### Alternative 4: Focus on OCR Quality Metrics
- Measure OCR quality directly (CER, WER against GPT-4V extraction)
- Correlate with Track B performance
- **Limitation**: Indirect measurement

## Recommended Implementation Plan

### Phase 1: Subset Human Annotation (n=200)
1. Select same 200 samples used in large_scale experiment
2. Create annotation interface (simple web form)
3. Manually transcribe questions + options
4. Store in JSON format alongside existing data

### Phase 2: 4-Track Experiment on Annotated Subset
```
Track A:  Image → Gemini Flash
Track B1: Gemini OCR Text → Gemini Flash  
Track B2: Human Text → Gemini Flash (NEW)
Track C:  Image + Gemini OCR → Gemini Flash
```

### Phase 3: Delta Analysis
- Compute Δ(B2-B1) = Pure OCR quality impact
- Stratify by region to identify language-specific OCR issues

## Data Schema for Human Transcription

```json
{
  "index": 7330,
  "nation": "South Korea",
  "task": "chemistry", 
  "human_transcription": {
    "question_text": "문 4. 샤를의 법칙을 옳게 표현한 식은?",
    "context": "(단, V, P, T, n은 각각 이상기체의 부피, 압력, 절대온도, 몰수이다)",
    "options": {
      "A": "V = 상수/P",
      "B": "V = 상수×n", 
      "C": "V = 상수×T",
      "D": "V = 상수×P"
    },
    "annotator": "human",
    "timestamp": "2026-01-20"
  }
}
```

## Alternative: Use OCR Output as Proxy

Given time constraints, we can:
1. Use the **existing OCR outputs** from our 200-sample experiment
2. **Manually verify** a subset (e.g., 50 samples) for OCR quality
3. Report OCR Character Error Rate (CER) as indirect measure

### OCR Quality Assessment Script
```python
# Compare OCR output quality across regions
# Metric: Character-level accuracy (manual spot check)
# Hypothesis: CER correlates with Track B performance
```

## Conclusion

**Recommendation**: Implement Alternative 1 (Subset Human Annotation) for rigorous causal analysis, OR Alternative 2 (High-Quality OCR Baseline) for faster iteration.

For NeurIPS submission, we should:
1. Acknowledge the limitation of not having oracle text
2. Propose future work direction for 4-track protocol
3. Use OCR quality metrics as indirect evidence

## Status: DESIGN COMPLETE
Next step: Implement OCR quality assessment on existing 200 samples
