# Prompt Sensitivity and Consistency in Local LLM Sentiment Classification

## Overview

This exploratory project compares VADER with Qwen2.5-1.5B running locally through Ollama for binary sentiment classification.

Three LLM prompting conditions are evaluated:
- Zero-shot
- Few-shot with six demonstrations
- Chain-of-thought (CoT)

The study measures classification performance and label consistency across repeated identical prompts.

## Research Questions

1. How does prompting strategy affect sentiment classification accuracy?
2. How do methods perform on sampled tweets and curated challenge sentences?
3. How consistent are predicted labels across three repeated runs?

## Dataset

The evaluation dataset contains 100 sentences.

| Group | Positive | Negative | Total |
|---|---:|---:|---:|
| Sampled tweets | 30 | 30 | 60 |
| Curated challenge sentences | 20 | 20 | 40 |
| Total | 50 | 50 | 100 |

Tweets were sampled from NLTK's Twitter Samples corpus. Challenge sentences were manually constructed to include negation, sarcasm, slang, and mixed sentiment. Their labels represent the intended sentiment.

The dataset-generation script uses Python random seed 42.

The CSV retains two original category names:
- `clear`: sampled tweets
- `ambiguous`: curated challenge sentences

These categories do not establish independently validated difficulty levels. The groups differ in both source and linguistic characteristics.

The repetition subset contains 30 sentences: 15 sampled tweets and 15 curated challenge sentences.

Use the supplied `dataset.csv` and `instability_subset_ids.csv` to reproduce the reported analysis.

## Methods

### VADER

VADER is a deterministic rule-based baseline. Compound scores are mapped to binary labels:
- Score ≥ 0: Positive
- Score < 0: Negative

This is the study's binary mapping rather than VADER's usual three-class decision rule.

### Local LLM

- Model tag: `qwen2.5:1.5b`
- Observed short model ID: `65ec06548149`
- Interface: Ollama local `/api/generate` endpoint
- Temperature: 0.7
- Generation seed: not explicitly set
- Other generation options: not explicitly set

**Zero-shot:** requests a single Positive or Negative label.

**Few-shot:** provides six demonstrations, with three positive and three negative examples. Revised demonstrations are separate from the evaluation sentences.

**CoT:** requests step-by-step analysis followed by `Final: Positive` or `Final: Negative`.

## Experimental Design

Each LLM condition includes:
- Run 1: all 100 sentences
- Run 2: the 30-sentence repetition subset
- Run 3: the same 30-sentence subset

The final dataset contains 160 records per prompting condition, totalling 480 LLM records. VADER contributes 100 predictions and was not repeated.

Main performance metrics use run 1 only. Label instability uses all three runs on the repetition subset.

The final records combine existing zero-shot and CoT responses with revised few-shot responses; they do not represent one newly collected experiment.

## Corrections to the Initial Experiment

Two original few-shot demonstrations appeared verbatim in the evaluation dataset. They were replaced with a set of six balanced demonstrations, and the few-shot condition was rerun.

Original few-shot responses remain in `results_llm.csv` for traceability but are excluded from the final analysis.

The corrected parser:
1. Restores saved newline separators.
2. Uses the last explicit `Final:` line when present.
3. Otherwise accepts only a standalone Positive or Negative response.
4. Returns `Unclear` for unsupported or malformed responses.

All retained responses were reparsed consistently. This changed the CoT prediction for sentence ID 31, run 1, from Negative to Positive.

Invalid predictions remain in the corrected evaluation instead of being excluded.

## Evaluation Metrics

- **Accuracy:** correct predictions divided by all evaluated examples, including invalid predictions in the denominator.
- **Macro-F1:** the unweighted mean of Positive and Negative F1. Invalid predictions count as missed predictions for their true class; `Unclear` is not a third class in the macro average.
- **Invalid-output rate:** the proportion of responses that cannot be parsed into a valid binary label.
- **Label instability:** the percentage of repeated sentences whose parsed label changes across three runs.

Invalid labels are included when checking label changes. Sentences with any invalid output are also reported separately.

Zero observed instability does not guarantee determinism or correctness.

## Final Results

### Main performance: run 1, all 100 sentences

| Method | Accuracy | Macro-F1 | Invalid outputs |
|---|---:|---:|---:|
| VADER | 75.0% | 0.743 | 0/100 |
| Zero-shot | 88.0% | 0.880 | 0/100 |
| Few-shot | 82.0% | 0.824 | 1/100 |
| CoT | 84.0% | 0.837 | 0/100 |

### Accuracy by dataset group

| Method | Sampled tweets (n=60) | Curated challenge sentences (n=40) |
|---|---:|---:|
| VADER | 81.7% | 65.0% |
| Zero-shot | 90.0% | 85.0% |
| Few-shot | 80.0% | 85.0% |
| CoT | 85.0% | 82.5% |

### Label instability: three runs on 30 sentences

| Condition | Sentences with label changes | Instability |
|---|---:|---:|
| Zero-shot | 0/30 | 0.0% |
| Few-shot | 0/30 | 0.0% |
| CoT | 5/30 | 16.7% |

Few-shot returned Neutral for sentence ID 2 in all three runs. These responses were parsed as `Unclear`, illustrating that consistent responses can still be invalid.

VADER was not repeated because it is deterministic by design.

## Interpretation

Zero-shot achieved the highest observed overall accuracy. Neither the revised few-shot prompt nor CoT improved overall accuracy over zero-shot.

Few-shot matched zero-shot on curated challenge sentences but performed worse on sampled tweets.

CoT showed the highest observed label instability. Some responses also contained contradictions between the explicit label and explanation.

These findings apply to this model, dataset, and prompt set. They do not establish that one strategy is universally better.

## Repository Files

| File | Purpose |
|---|---|
| `dataset.csv` | Fixed evaluation dataset |
| `instability_subset_ids.csv` | Fixed repetition subset |
| `prompts.py` | Current prompts and corrected parser |
| `llm_client.py` | Ollama request wrapper |
| `results_llm.csv` | Original responses, including superseded few-shot results |
| `results_few_shot_v2.csv` | Revised few-shot responses |
| `results_llm_final.csv` | Combined and reparsed final LLM records |
| `results_vader.csv` | VADER predictions |
| `1_data_prep.py` | Dataset and subset generation |
| `2_run_experiment.py` | LLM experiment runner |
| `3_run_vader.py` | VADER baseline runner |
| `5_merge_results.py` | Response validation, merging, and reparsing |
| `6_analyze_final.py` | Corrected metrics and figures |
| `requirements.txt` | Recorded direct dependency versions |
| `final_analysis/` | Final tables and figures |

`2_run_experiment.py` is currently configured for few-shot only, saving to `results_few_shot_v2.csv`.

`test_ollama_sentiment.py` is a connectivity smoke test. Its three-class prompt is not part of the binary experiment.

## Reproduce the Reported Analysis

Open a terminal in the repository folder. Install dependencies, preferably in a separate Python environment:

```bash
python -m pip install -r requirements.txt
```

With the supplied data and final prediction files present, run:

```bash
python 6_analyze_final.py
```

This validates the inputs and recreates the analysis. Ollama and new model calls are not required.

Outputs in `final_analysis/`:
- `metrics_final.csv`
- `stability_final.csv`
- `invalid_responses_final.csv`
- `errors_final.csv`
- `accuracy_final.png`
- `accuracy_final.pdf`
- `instability_final.png`
- `instability_final.pdf`

Rerunning the analysis replaces its outputs but does not modify input prediction files.

## Rebuild the Merged Results

If `results_llm_final.csv` does not already exist, run:

```bash
python 5_merge_results.py
python 6_analyze_final.py
```

The merge script requires the original responses, revised few-shot responses, fixed dataset, repetition subset, and current parser.

It refuses to overwrite an existing final results file. Preserve that file before rebuilding.

## Collect New Model Responses

New model calls are not required to reproduce the reported analysis.

For a new experiment:
1. Work in a separate project copy.
2. Keep the dataset, subset, and chosen prompt versions fixed.
3. Ensure Ollama is running and the intended model is available.
4. Select the desired conditions in `2_run_experiment.py`.
5. Set a new, unused results filename.
6. Run `python 2_run_experiment.py`.

The runner resumes using sentence ID, condition, and run number. It does not detect changes to prompts, models, or settings. Use a new output filename after changing any of these inputs.

All three conditions produce 480 records; the current few-shot-only configuration produces 160.

Generation is stochastic, so new responses and scores may differ.

## Recorded Environment

Versions reported during final documentation:
- Python: 3.13.3
- Ollama: 0.34.1
- requests: 2.34.2
- nltk: 3.10.3
- vaderSentiment: 3.3.2
- pandas: 3.0.5
- scikit-learn: 1.9.0
- matplotlib: 3.11.1

These are reported environment versions, not a verified historical snapshot for every earlier model call. Requirements pin direct dependencies rather than the entire environment.

## Limitations

- Small evaluation dataset of 100 sentences.
- One model and one prompt template per condition.
- Consistency measured using only three runs on 30 sentences.
- Dataset groups differ in source and linguistic characteristics.
- Challenge labels reflect intended sentiment without established independent annotator agreement.
- Some tweets require missing context or contain multilingual text.
- Binary classification excludes neutral and mixed-sentiment labels.
- Few-shot revisions followed discovery of evaluation overlap; this is an exploratory corrected study.
- Earlier model calls lack a complete historical environment manifest.
- Differences are descriptive; no statistical-significance claim is made.
- Generated explanations do not establish the model's internal reasoning.
