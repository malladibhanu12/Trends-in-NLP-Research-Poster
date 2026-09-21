\## Prompt Sensitivity and Consistency in Local LLM Sentiment Classification



\## Overview



This exploratory NLP project compares VADER with Qwen2.5-1.5B

running locally through Ollama for binary sentiment classification.



Three LLM prompting conditions are evaluated:

\- Zero-shot

\- Few-shot with six demonstrations

\- Chain-of-thought (CoT) prompting



The study measures classification performance and label consistency

across repeated identical prompts.



\## Research Questions



1\. How does prompting strategy affect sentiment classification accuracy?

2\. How do methods perform on sampled tweets and curated challenge sentences?

3\. How consistent are the predicted labels across three repeated runs?



\## Dataset



The evaluation dataset contains 100 sentences:



| Group | Positive | Negative | Total |

|---|---:|---:|---:|

| Sampled tweets | 30 | 30 | 60 |

| Curated challenge sentences | 20 | 20 | 40 |

| Total | 50 | 50 | 100 |



The tweets were sampled from NLTK's Twitter Samples corpus.

The challenge sentences were manually constructed to include negation,

sarcasm, slang, and mixed sentiment. Their labels represent the

intended sentiment.



The dataset-generation script uses Python random seed 42.



The CSV category names are:

\- `clear`: sampled tweets

\- `ambiguous`: curated challenge sentences



These names are retained for compatibility with the original code.

They do not establish independently validated difficulty levels.

The groups differ in both source and linguistic characteristics.



The repetition subset contains 30 sentences:

15 sampled tweets and 15 curated challenge sentences.



Use the supplied `dataset.csv` and `instability\_subset\_ids.csv`

to reproduce the reported analysis. Do not regenerate them first.



\## Methods



\### VADER



VADER is used as a deterministic rule-based baseline.



Its compound score is mapped to binary labels:

\- Score >= 0: Positive

\- Score < 0: Negative



This is the study's binary mapping, not VADER's usual three-class

positive/neutral/negative decision rule.



\### Local LLM



\- Model tag: `qwen2.5:1.5b`

\- Observed short model ID: `65ec06548149`

\- Interface: Ollama local `/api/generate` endpoint

\- Temperature: 0.7 for all main and repeated LLM calls

\- Generation seed: not explicitly set in the request

\- Other generation options: not explicitly set in the request



Zero-shot requests a single Positive or Negative label.



Few-shot provides six demonstrations: three positive and three negative.

The revised demonstrations are separate from the evaluation sentences.



CoT requests step-by-step analysis followed by an explicit

`Final: Positive` or `Final: Negative` answer.

The model does not always follow the requested reasoning or output order.



\## Experimental Design



Each LLM condition has:

\- 100 main predictions: run 1

\- 30 additional predictions: run 2

\- 30 additional predictions: run 3



The final analysis contains 480 LLM records in total:

160 per prompting condition.



VADER has 100 predictions and was not repeated.



Main performance metrics use run 1 only.

Runs 2 and 3 are used for label-consistency analysis.



The 480 final records include existing zero-shot and CoT responses

and the revised few-shot responses. They are not all from one new run.



\## Corrections to the Initial Experiment



Two original few-shot demonstrations appeared verbatim in the

evaluation dataset. The demonstration set was replaced with six

balanced examples, and the few-shot condition was rerun.



The old few-shot results remain in `results\_llm.csv` for traceability,

but are excluded from the final analysis.



The parser was corrected to:

1\. Restore saved newline separators.

2\. Inspect the last explicit `Final:` line, if present.

3\. Otherwise accept only a standalone Positive or Negative response.

4\. Return `Unclear` for unsupported or malformed answers.



All retained responses were reparsed consistently.

This changed the CoT prediction for sentence ID 31, run 1,

from Negative to Positive.



The original analysis excluded invalid predictions.

The corrected analysis retains them as unsuccessful classifications.



\## Evaluation Metrics



\### Accuracy



Correct predictions divided by all evaluated examples.

Invalid predictions remain in the denominator.



\### Macro-F1



The unweighted mean of F1 for Positive and Negative.

Invalid predictions are represented as `Unclear` and count as

missed predictions for their true class.

`Unclear` is not included as a third target class in the macro average.



\### Invalid-output rate



The proportion of responses that cannot be parsed into a valid

Positive or Negative label under the fixed parsing rule.



\### Label instability



The percentage of the 30 repeated sentences for which the parsed

label differs across the three runs.



Invalid labels are included when checking label changes.

Sentences with any invalid output are also reported separately.



Zero observed instability does not imply guaranteed determinism

or correct predictions.



\## Final Results



\### Main performance: run 1, all 100 sentences



| Method | Accuracy | Macro-F1 | Invalid outputs |

|---|---:|---:|---:|

| VADER | 75.0% | 0.743 | 0/100 |

| Zero-shot | 88.0% | 0.880 | 0/100 |

| Few-shot | 82.0% | 0.824 | 1/100 |

| CoT | 84.0% | 0.837 | 0/100 |



\### Accuracy by dataset group



| Method | Sampled tweets (n=60) | Challenge sentences (n=40) |

|---|---:|---:|

| VADER | 81.7% | 65.0% |

| Zero-shot | 90.0% | 85.0% |

| Few-shot | 80.0% | 85.0% |

| CoT | 85.0% | 82.5% |



\### Label instability: three runs on 30 sentences



| Condition | Sentences with label changes | Instability |

|---|---:|---:|

| Zero-shot | 0/30 | 0.0% |

| Few-shot | 0/30 | 0.0% |

| CoT | 5/30 | 16.7% |



Few-shot returned Neutral for sentence ID 2 in all three runs.

These responses were parsed as Unclear. This illustrates that a

consistent response can still be invalid for the task.



\## Interpretation



Zero-shot achieved the highest observed overall accuracy.

Neither the revised few-shot prompt nor the CoT prompt improved

overall accuracy over zero-shot in this experiment.



Few-shot matched zero-shot on challenge sentences but performed

worse on sampled tweets.



CoT showed the most observed label instability.

Some CoT responses also contained contradictions between their

explicit label and explanation.



These findings concern this model, dataset, and prompt set.

They do not establish that one prompting strategy is universally better.



\## Important Files



| File | Purpose |

|---|---|

| `dataset.csv` | Fixed 100-sentence evaluation dataset |

| `instability\_subset\_ids.csv` | Fixed 30-sentence repetition subset |

| `prompts.py` | Current prompts and corrected parser |

| `llm\_client.py` | Local Ollama request wrapper |

| `results\_llm.csv` | Original responses, including superseded few-shot results |

| `results\_few\_shot\_v2.csv` | Revised few-shot responses |

| `results\_llm\_final.csv` | Combined and reparsed final LLM records |

| `results\_vader.csv` | VADER baseline predictions |

| `5\_merge\_results.py` | Validate, combine, and reparse responses |

| `6\_analyze\_final.py` | Generate corrected metrics and figures |

| `requirements.txt` | Recorded direct dependency versions |

| `final\_analysis/` | Corrected analysis outputs |



`1\_data\_prep.py` generates the dataset and repetition subset.

It is not needed when using the supplied data.



`2\_run\_experiment.py` is currently configured to run only few-shot

and save to `results\_few\_shot\_v2.csv`.



`3\_run\_vader.py` runs the VADER baseline and writes `results\_vader.csv`.



`4\_analyze\_results.py` and the old `accuracy\_chart.png` are legacy

artifacts. Do not use them for the final results.



`test\_ollama\_sentiment.py` is a connectivity smoke test.

Its three-class prompt is not part of the binary experiment.



\## Reproduce the Reported Analysis Without Model Calls



Open a terminal in the project folder.



Install the dependencies, preferably in a separate Python environment:



```bash

python -m pip install -r requirements.txt

```



With the supplied final results present, run:



```bash

python 6\_analyze\_final.py

```



This validates the inputs and recreates the tables and figures.

Ollama is not needed for this analysis-only workflow.



Outputs are saved inside `final\_analysis/`:

\- `metrics\_final.csv`

\- `stability\_final.csv`

\- `invalid\_responses\_final.csv`

\- `errors\_final.csv`

\- `accuracy\_final.png`

\- `accuracy\_final.pdf`

\- `instability\_final.png`

\- `instability\_final.pdf`



Running the analysis again replaces these analysis outputs,

but does not modify the input prediction files.



\## Rebuild the Merged Results



If `results\_llm\_final.csv` does not already exist, run:



```bash

python 5\_merge\_results.py

python 6\_analyze\_final.py

```



The merge script requires the original results, revised few-shot

results, fixed dataset, repetition subset, and current parser.



It refuses to overwrite an existing `results\_llm\_final.csv`.

Preserve an existing final file before rebuilding it.



\## Collect New Model Responses



New model runs are not required to verify the reported metrics.



For a new experiment:

1\. Work in a separate copy of the project.

2\. Keep the dataset, subset, and chosen prompt versions fixed.

3\. Ensure Ollama is running and the intended model is available.

4\. Select the desired conditions in `2\_run\_experiment.py`.

5\. Set a new, unused results filename before running.

6\. Run `python 2\_run\_experiment.py`.



The runner resumes using sentence ID, condition, and run number.

It does not automatically detect changes to prompts, models, or settings.

Never reuse a results filename after changing those inputs.



A new full run with all three conditions produces 480 records.

The currently configured few-shot-only run produces 160 records.



Generation is stochastic, so newly collected responses and scores

may differ from the supplied results.



\## Recorded Environment



Versions reported during final documentation:

\- Python: 3.13.3

\- Ollama: 0.34.1

\- requests: 2.34.2

\- nltk: 3.10.3

\- vaderSentiment: 3.3.2

\- pandas: 3.0.5

\- scikit-learn: 1.9.0

\- matplotlib: 3.11.1



These are the currently recorded versions, not a verified historical

environment snapshot for every earlier model call.

The requirements file pins direct dependencies, not the entire environment.



\## Limitations



\- Small evaluation dataset: 100 sentences.

\- Only one local model and one prompt template per condition.

\- Only three runs on a 30-sentence subset for consistency.

\- Dataset groups differ in source as well as language characteristics.

\- Challenge labels reflect intended sentiment; independent annotator

&#x20; agreement has not been established.

\- Some tweets require missing context or contain multilingual text.

\- The binary task excludes neutral and mixed-sentiment labels.

\- Few-shot revisions followed discovery of evaluation overlap;

&#x20; this is an exploratory corrected study, not a preregistered experiment.

\- Earlier model calls lack a complete historical environment manifest.

\- Reported differences are descriptive; no statistical-significance

&#x20; claim is made.

\- Generated explanations are not proof of a model's internal reasoning.

