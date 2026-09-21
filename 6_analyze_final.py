"""
Analyze the corrected sentiment-classification experiment.

Main accuracy uses run 1 only.
Invalid predictions remain in the evaluation denominator.
Macro-F1 averages over the two target classes.
Instability measures label changes across three runs on 30 sentences.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FOLDER = Path(__file__).resolve().parent
OUTPUT = FOLDER / "final_analysis"

LABELS = ["Positive", "Negative"]
CONDITIONS = ["zero_shot", "few_shot", "cot"]

METHOD_NAMES = {
    "zero_shot": "Zero-shot",
    "few_shot": "Few-shot",
    "cot": "CoT",
}

GROUP_NAMES = {
    "clear": "Sampled tweets",
    "ambiguous": "Curated challenge sentences",
}


def read_csv(filename):
    """Read a CSV without converting empty responses into NaN."""
    return pd.read_csv(FOLDER / filename, keep_default_na=False)


def validate_results(dataset, subset, llm, vader):
    """Require complete, unique records matching the dataset."""
    if len(dataset) != 100 or dataset["id"].nunique() != 100:
        raise ValueError("Expected 100 unique dataset IDs.")

    if not dataset["gold_label"].isin(LABELS).all():
        raise ValueError("Unexpected gold labels.")

    if not dataset["category"].isin(GROUP_NAMES).all():
        raise ValueError("Unexpected dataset categories.")

    subset_ids = set(subset["id"])
    dataset_ids = set(dataset["id"])

    if len(subset) != 30 or len(subset_ids) != 30:
        raise ValueError("Expected 30 unique repetition-subset IDs.")

    if not subset_ids.issubset(dataset_ids):
        raise ValueError("Unknown IDs in the repetition subset.")

    keys = ["id", "condition", "run_number"]

    if llm.duplicated(keys).any():
        raise ValueError("Duplicate LLM records.")

    expected = {
        (sentence_id, condition, run)
        for sentence_id in dataset_ids
        for condition in CONDITIONS
        for run in (
            (1, 2, 3) if sentence_id in subset_ids else (1,)
        )
    }

    actual = set(llm[keys].itertuples(index=False, name=None))

    if actual != expected:
        raise ValueError(
            f"LLM records: {len(expected - actual)} missing, "
            f"{len(actual - expected)} unexpected."
        )

    if len(vader) != 100 or vader["id"].duplicated().any():
        raise ValueError("Expected 100 unique VADER records.")

    if set(vader["id"]) != dataset_ids:
        raise ValueError("VADER IDs do not match the dataset.")

    reference = dataset.set_index("id")

    for name, frame in [("LLM", llm), ("VADER", vader)]:
        for column in ["text", "gold_label", "category"]:
            expected_values = frame["id"].map(reference[column])
            if not frame[column].eq(expected_values).all():
                raise ValueError(f"{name}: mismatched {column}.")

    print("Validation passed: complete records matching the dataset.")


def calculate_metrics(frame, prediction_column):
    """Count invalid predictions as errors without dropping rows."""
    gold = frame["gold_label"]
    predictions = frame[prediction_column]
    valid = predictions.isin(LABELS)

    # Retain invalid answers as a non-target prediction.
    # They contribute false negatives for their true target class.
    predictions = predictions.where(valid, "Unclear")

    return {
        "n": len(frame),
        "correct": int(gold.eq(predictions).sum()),
        "accuracy_pct": 100 * accuracy_score(gold, predictions),
        "macro_f1": f1_score(
            gold,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "invalid_count": int((~valid).sum()),
        "invalid_pct": 100 * (~valid).mean(),
    }


def main():
    dataset = read_csv("dataset.csv")
    subset = read_csv("instability_subset_ids.csv")
    llm = read_csv("results_llm_final.csv")
    vader = read_csv("results_vader.csv")

    validate_results(dataset, subset, llm, vader)

    OUTPUT.mkdir(exist_ok=True)
    main_llm = llm[llm["run_number"] == 1].copy()

    methods = {"VADER": (vader, "predicted_label")}

    for condition in CONDITIONS:
        frame = main_llm[main_llm["condition"] == condition]
        methods[METHOD_NAMES[condition]] = (frame, "parsed_label")

    # Main and category-specific metrics.
    metric_rows = []

    for method, (frame, prediction_column) in methods.items():
        groups = [("Overall", frame)]

        for category, display_name in GROUP_NAMES.items():
            groups.append(
                (display_name, frame[frame["category"] == category])
            )

        for group_name, group in groups:
            metric_rows.append({
                "method": method,
                "group": group_name,
                **calculate_metrics(group, prediction_column),
            })

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUTPUT / "metrics_final.csv", index=False)

    print("\nMAIN RESULTS: RUN 1 ONLY")
    overall = metrics[metrics["group"] == "Overall"]
    print(overall.to_string(index=False, float_format="%.3f"))

    print("\nRESULTS BY DATASET GROUP")
    grouped = metrics[metrics["group"] != "Overall"]
    print(grouped.to_string(index=False, float_format="%.3f"))

    # Repeat-run stability: include invalid labels but report them separately.
    stability_rows = []
    subset_ids = set(subset["id"])

    for condition in CONDITIONS:
        repeated = llm[
            (llm["condition"] == condition)
            & llm["id"].isin(subset_ids)
        ].copy()

        repeated["evaluation_label"] = repeated["parsed_label"].where(
            repeated["parsed_label"].isin(LABELS), "Unclear"
        )
        repeated["invalid"] = ~repeated["parsed_label"].isin(LABELS)

        unique_labels = repeated.groupby("id")["evaluation_label"].nunique()
        any_invalid = repeated.groupby("id")["invalid"].any()
        changed = int((unique_labels > 1).sum())

        stability_rows.append({
            "method": METHOD_NAMES[condition],
            "sentences": len(unique_labels),
            "sentences_with_label_changes": changed,
            "instability_pct": 100 * changed / len(unique_labels),
            "sentences_with_any_invalid": int(any_invalid.sum()),
        })

    stability = pd.DataFrame(stability_rows)
    stability.to_csv(OUTPUT / "stability_final.csv", index=False)

    print("\nLABEL INSTABILITY: THREE RUNS PER SENTENCE")
    print(stability.to_string(index=False, float_format="%.2f"))
    print("VADER was not repeated; it is deterministic by design.")

    # Save all invalid LLM outputs for inspection.
    invalid = llm[~llm["parsed_label"].isin(LABELS)]
    invalid.to_csv(OUTPUT / "invalid_responses_final.csv", index=False)

    # Save every run-1 error, including invalid predictions.
    error_frames = []

    for method, (frame, prediction_column) in methods.items():
        errors = frame[
            frame[prediction_column] != frame["gold_label"]
        ].copy()

        errors["method"] = method
        errors["prediction"] = errors[prediction_column]

        if "raw_response" not in errors.columns:
            errors["raw_response"] = ""

        error_frames.append(errors[[
            "method", "id", "text", "category",
            "gold_label", "prediction", "raw_response",
        ]])

    pd.concat(error_frames, ignore_index=True).to_csv(
        OUTPUT / "errors_final.csv", index=False
    )

    # Accuracy chart: full denominators, not valid responses only.
    plt.rcParams.update({"font.size": 12})
    method_order = list(methods)
    positions = list(range(len(method_order)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 6))

    for offset, (group_name, color) in enumerate([
        ("Sampled tweets", "#2878B5"),
        ("Curated challenge sentences", "#E58B24"),
    ]):
        values = [
            metrics.loc[
                (metrics["method"] == method)
                & (metrics["group"] == group_name),
                "accuracy_pct",
            ].iloc[0]
            for method in method_order
        ]

        x_values = [
            position + (offset - 0.5) * width
            for position in positions
        ]

        bars = ax.bar(
            x_values, values, width,
            label=group_name, color=color,
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=10)

    ax.set_xticks(positions)
    ax.set_xticklabels(method_order)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Sentiment accuracy by method and dataset group")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUTPUT / "accuracy_final.png", dpi=300)
    fig.savefig(OUTPUT / "accuracy_final.pdf")
    plt.close(fig)

    # Label-instability chart.
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        stability["method"],
        stability["instability_pct"],
        color=["#2878B5", "#E58B24", "#649E52"],
    )

    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_ylim(
        0, max(25, float(stability["instability_pct"].max()) + 10)
    )
    ax.set_ylabel("Sentences with label changes (%)")
    ax.set_title("Label instability: 30 sentences, three runs each")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(OUTPUT / "instability_final.png", dpi=300)
    fig.savefig(OUTPUT / "instability_final.pdf")
    plt.close(fig)

    print(f"\nTables and charts saved in:\n{OUTPUT}")
    print("Input files were not changed.")
    print("Running this script again replaces only its analysis outputs.")


if __name__ == "__main__":
    main()