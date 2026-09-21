"""
STEP 3: Run the VADER rule-based baseline on the full dataset.

VADER is fully deterministic, so it only needs to run once per sentence
(no repeats needed for an instability check -- by construction, instability
is always 0%).

Citation: Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious
Rule-based Model for Sentiment Analysis of Social Media Text. Eighth
International AAAI Conference on Weblogs and Social Media (ICWSM-14).

This script runs in a few seconds.
"""

import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

with open("dataset.csv", newline="", encoding="utf-8") as f:
    dataset = list(csv.DictReader(f))

results = []
for row in dataset:
    scores = analyzer.polarity_scores(row["text"])
    compound = scores["compound"]
    # Binary mapping: compound >= 0 -> Positive, else Negative
    # (Note: VADER's typical 3-way threshold uses a neutral band around 0;
    # since our task is binary, we use 0 as the single decision boundary.)
    predicted_label = "Positive" if compound >= 0 else "Negative"

    results.append({
        "id": row["id"],
        "text": row["text"],
        "gold_label": row["gold_label"],
        "category": row["category"],
        "compound_score": compound,
        "predicted_label": predicted_label,
    })

with open("results_vader.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "text", "gold_label", "category", "compound_score", "predicted_label"])
    writer.writeheader()
    writer.writerows(results)

correct = sum(1 for r in results if r["predicted_label"] == r["gold_label"])
print(f"VADER results saved to results_vader.csv")
print(f"Quick accuracy check: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
