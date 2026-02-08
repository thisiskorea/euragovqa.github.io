# Datasheet for EuraGovExam

Following the framework proposed by Gebru et al. (2021), "Datasheets for Datasets."

---

## Motivation

### For what purpose was the dataset created?

EuraGovExam was created to evaluate Vision-Language Models (VLMs) on authentic, multilingual, document-like examination images. Unlike existing benchmarks that use synthetically rendered text or clean academic problems, EuraGovExam captures the challenges of real-world document understanding: complex layouts, scanning artifacts, diverse writing systems, and jurisdiction-specific knowledge requirements.

The dataset addresses a critical gap: while VLMs have achieved impressive results on standard benchmarks, their performance on real government documents remains poorly understood. Civil service exams provide a unique testbed because they:
1. Require reading scanned document images (not rendered text)
2. Test jurisdiction-specific knowledge (not universal facts)
3. Have official answer keys (ensuring annotation quality)
4. Represent high-stakes applications (government employment)

### Who created the dataset and on behalf of which entity?

[Anonymous for review - will be disclosed upon acceptance]

### Who funded the creation of the dataset?

[To be disclosed upon acceptance]

---

## Composition

### What do the instances that comprise the dataset represent?

Each instance represents a single civil service examination question, consisting of:
- **Image**: A scanned or photographed image of the exam question as it appeared in the original document
- **Metadata**: Nation, subject domain (task), year, script type, difficulty level
- **Ground Truth**: The official correct answer from published answer keys

### How many instances are there in total?

The dataset contains **8,000+** examination questions distributed across:

| Region | Questions | Primary Script | Languages |
|--------|-----------|----------------|-----------|
| Korea | ~2,000 | Hangul | Korean |
| Japan | ~1,500 | Japanese (Kanji + Hiragana + Katakana) | Japanese |
| Taiwan | ~1,500 | Traditional Chinese | Mandarin |
| India | ~1,500 | Devanagari, Latin | Hindi, English |
| EU | ~1,500 | Latin | English, French, German, etc. |

### Does the dataset contain all possible instances or is it a sample?

The dataset is a sample of publicly available civil service examinations. We prioritized:
1. **Recent exams** (2015-2024) to minimize data contamination in LLM training
2. **Diverse subjects** covering 17 domains
3. **Image quality variance** to test robustness

### What data does each instance consist of?

```json
{
  "id": "korea_2023_law_042",
  "nation": "korea",
  "task": "law",
  "script": "hangul",
  "year": 2023,
  "img": "images/korea/2023/law/042.png",
  "correct_answer": "3",
  "num_choices": 5,
  "difficulty": "medium",
  "exam_type": "grade_9",
  "ocr_text": null,
  "oracle_text": null
}
```

### Is there a label or target associated with each instance?

Yes, each instance has a `correct_answer` field containing the official answer (typically 1-5 for multiple choice questions). These answers are sourced from:
- Official answer keys published by examination authorities
- Government websites
- Official exam preparation materials

### Is any information missing from individual instances?

Some optional fields may be missing:
- `ocr_text`: OCR-extracted text (available for subset)
- `oracle_text`: Human-transcribed text (available for ~500 instances)
- `difficulty`: Not available for all regions

### Are relationships between individual instances made explicit?

Questions from the same exam session share metadata (year, exam_type). No explicit relationships are encoded, but filtering by metadata enables structured analysis.

### Are there recommended data splits?

We recommend the following splits:

| Split | Purpose | Size | Selection Criteria |
|-------|---------|------|-------------------|
| Test | Main evaluation | 5,000 | Balanced across regions/tasks |
| Val | Hyperparameter tuning | 1,500 | Temporal split (2023 exams) |
| Hard | Challenge subset | 1,500 | Questions most models fail |

### Are there any errors, sources of noise, or redundancies?

**Known issues:**
1. **Scanning quality variance**: Some images have low resolution or scanning artifacts
2. **Cropping inconsistency**: Question boundaries may not be perfectly aligned
3. **Near-duplicates**: Some questions test similar concepts (identified via perceptual hashing)
4. **Answer key errors**: Rare (<0.1%) official answer key errors have been identified and corrected

### Is the dataset self-contained?

Yes. All images and metadata are included. No external data is required for evaluation.

---

## Collection Process

### How was the data associated with each instance acquired?

Data was collected from:
1. **Official government websites**: Published past exam papers
2. **Public exam preparation portals**: Licensed or public domain materials
3. **Official publications**: Printed exam booklets (scanned)

### What mechanisms or procedures were used to collect the data?

1. **Web scraping**: Automated collection from official sources
2. **Manual scanning**: Physical exam papers digitized
3. **API access**: Where available through official channels

### If the dataset is a sample, what was the sampling strategy?

We employed stratified sampling to ensure:
- Balanced representation across regions
- Coverage of all 17 subject domains
- Temporal diversity (2015-2024)
- Difficulty variance

### Who was involved in the data collection process?

[To be disclosed - graduate student researchers with domain expertise in each region]

### Over what timeframe was the data collected?

Data collection occurred from January 2024 to December 2024.

### Were any ethical review processes conducted?

Yes. The research was reviewed by [institutional ethics board - anonymized]. Key considerations:
- All data is from publicly available sources
- No personally identifiable information (PII) is included
- Questions test general knowledge, not personal information

---

## Preprocessing/Cleaning/Labeling

### Was any preprocessing/cleaning/labeling of the data done?

**Preprocessing:**
1. Image normalization (consistent DPI, format conversion to PNG)
2. Border cropping (removal of page margins)
3. Quality filtering (removal of illegible images)

**Cleaning:**
1. PII removal (redaction of any names, ID numbers)
2. Duplicate detection (perceptual hashing)
3. Answer key validation (cross-checking with multiple sources)

**Labeling:**
1. Subject domain classification (manual by domain experts)
2. Difficulty estimation (based on historical pass rates where available)
3. Script type identification (automatic + manual verification)

### Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?

Yes. Original scanned images are preserved. Preprocessing is reversible.

### Is the software that was used to preprocess/clean/label the data available?

Yes. Preprocessing scripts will be released with the dataset.

---

## Uses

### Has the dataset been used for any tasks already?

The dataset was used for the experiments reported in our paper:
- VLM evaluation (23 models)
- Regional bottleneck analysis
- Diagnostic protocol development

### Is there a repository that links to any or all papers or systems that use the dataset?

[To be created upon publication]

### What (other) tasks could the dataset be used for?

1. **OCR benchmarking**: Evaluating multilingual OCR systems
2. **Document layout analysis**: Detecting tables, diagrams, equations
3. **Knowledge extraction**: Assessing domain-specific knowledge
4. **Multilingual NLP**: Cross-lingual transfer studies
5. **Educational AI**: Automated exam grading/generation

### Is there anything about the composition of the dataset or the way it was collected that might impact future uses?

**Considerations:**
1. **Temporal scope**: Exams from 2015-2024 may become contaminated in future LLM training
2. **Regional bias**: Some regions have more data than others
3. **Subject balance**: Not all subjects are equally represented
4. **Language evolution**: Legal terminology may change over time

### Are there tasks for which the dataset should not be used?

1. **Cheating assistance**: The dataset should not be used to help candidates cheat on actual exams
2. **Commercial exam prep without license**: Respect original copyright holders
3. **Discriminatory applications**: Do not use to evaluate humans based on protected characteristics

---

## Distribution

### Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?

Yes. The dataset will be publicly available under CC BY-NC-SA 4.0 license.

### How will the dataset be distributed?

1. **HuggingFace Datasets**: Primary distribution (https://huggingface.co/datasets/EuraGovExam/EuraGovExam)
2. **GitHub**: Code, documentation, and leaderboard
3. **Paper supplementary**: Subset for review

### When will the dataset be distributed?

Upon paper acceptance.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license?

**License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)

**Rationale**:
- **BY**: Attribution required to acknowledge source
- **NC**: Non-commercial to respect original exam creators
- **SA**: Share-alike to ensure derivatives remain open

### Have any third parties imposed IP-based or other restrictions on the data?

Original exam materials are typically:
- Public domain (government works) or
- Fair use for research purposes

We have verified redistribution rights for each region's materials.

---

## Maintenance

### Who will be supporting/hosting/maintaining the dataset?

[Authors' institution - anonymized]

### How can the owner/curator/manager of the dataset be contacted?

[Email to be provided upon publication]

### Is there an erratum?

An erratum will be maintained on the GitHub repository.

### Will the dataset be updated?

**Planned updates:**
1. **Annual expansion**: New exam years added
2. **Error correction**: Community-reported issues
3. **Additional regions**: Southeast Asia, Latin America (planned)

### Will older versions of the dataset continue to be supported?

Yes. Versioned releases will be maintained (v1.0, v1.1, etc.).

### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism?

Yes:
1. **GitHub Issues**: Report errors or request additions
2. **Pull Requests**: Contribute new data (with provenance)
3. **Leaderboard**: Submit model evaluations

---

## Legal and Ethical Considerations

### Were any ethical considerations taken into account?

1. **Privacy**: No PII in any question
2. **Fairness**: Balanced regional representation
3. **Access**: Open access to promote research equity
4. **Consent**: Using publicly available government materials

### Is there any sensitive data in the dataset?

No. All questions test general knowledge. No personal, medical, financial, or other sensitive information is present.

### Does the dataset contain data that might be considered offensive?

Civil service exams are professionally curated and do not contain offensive content. Questions testing historical or political topics are presented factually.

### Does the dataset identify any subpopulations?

Questions may reference demographic groups in a factual, educational context (e.g., population statistics, government policies). No individual identification is possible.

---

## Additional Notes

### Key Statistics

| Metric | Value |
|--------|-------|
| Total questions | 8,000+ |
| Regions | 5 |
| Writing systems | 4 |
| Subject domains | 17 |
| Year range | 2015-2024 |
| Image resolution | 150-600 DPI |
| Answer format | Multiple choice (4-5 options) |

### Acknowledgments

We thank the examination authorities of Korea, Japan, Taiwan, India, and EU member states for making past examination materials publicly available. This research benefits from their commitment to transparency in government employment.

---

**Document Version**: 1.0
**Last Updated**: January 2025
