"""
Combine existing zero-shot/CoT responses with the new few-shot responses.

Validates completeness and dataset consistency before saving.
Original input files are never modified.
"""

import csv
from collections import Counter
from pathlib import Path

from prompts import parse_final_label


FOLDER = Path(__file__).resolve().parent
OUTPUT = FOLDER / "results_llm_final.csv"

FIELDS = [
    "id", "text", "gold_label", "category",
    "condition", "run_number", "raw_response", "parsed_label",
]


def read_csv(filename):
    """Read a CSV from the same folder as this script."""
    with (FOLDER / filename).open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        return list(csv.DictReader(file))


def main():
    dataset = read_csv("dataset.csv")
    subset = read_csv("instability_subset_ids.csv")
    original = read_csv("results_llm.csv")
    new_few_shot = read_csv("results_few_shot_v2.csv")

    dataset_by_id = {int(row["id"]): row for row in dataset}
    subset_ids = {int(row["id"]) for row in subset}

    if len(dataset) != 100 or len(dataset_by_id) != 100:
        raise ValueError("Expected 100 unique dataset IDs.")

    if len(subset) != 30 or len(subset_ids) != 30:
        raise ValueError("Expected 30 unique repetition-subset IDs.")

    if not subset_ids.issubset(dataset_by_id):
        raise ValueError("The repetition subset contains unknown IDs.")

    if any(row["condition"] != "few_shot" for row in new_few_shot):
        raise ValueError("The new file must contain only few_shot results.")

    # Discard old few-shot results from this combination only.
    # The original CSV remains untouched.
    combined = [
        row.copy()
        for row in original
        if row["condition"] in {"zero_shot", "cot"}
    ]
    combined.extend(row.copy() for row in new_few_shot)

    conditions = ("zero_shot", "few_shot", "cot")

    expected_keys = {
        (sentence_id, condition, run)
        for sentence_id in dataset_by_id
        for condition in conditions
        for run in (
            (1, 2, 3) if sentence_id in subset_ids else (1,)
        )
    }

    actual_keys = [
        (int(row["id"]), row["condition"], int(row["run_number"]))
        for row in combined
    ]

    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("Duplicate experiment records found.")

    missing = expected_keys - set(actual_keys)
    extra = set(actual_keys) - expected_keys

    if missing or extra:
        raise ValueError(
            f"Records do not match the experiment: "
            f"{len(missing)} missing, {len(extra)} unexpected."
        )

    changed = 0

    for row in combined:
        reference = dataset_by_id[int(row["id"])]

        for column in ("text", "gold_label", "category"):
            if row[column] != reference[column]:
                raise ValueError(
                    f"Dataset mismatch: ID {row['id']}, column {column}."
                )

        # The updated parser handles saved ' | ' newline separators.
        new_label = parse_final_label(row["raw_response"])

        if new_label != row["parsed_label"]:
            changed += 1
            print(
                f"Updated ID {row['id']}, {row['condition']}, "
                f"run {row['run_number']}: "
                f"{row['parsed_label']} -> {new_label}"
            )

        row["parsed_label"] = new_label

    combined.sort(
        key=lambda row: (
            int(row["id"]),
            conditions.index(row["condition"]),
            int(row["run_number"]),
        )
    )

    # Exclusive creation prevents overwriting an existing final file.
    with OUTPUT.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(
            {field: row[field] for field in FIELDS}
            for row in combined
        )

    print("\nValidation passed: no missing or duplicate records.")
    print(f"Saved {len(combined)} records to {OUTPUT.name}")
    print(f"Labels changed during reparsing: {changed}")

    counts = Counter(row["condition"] for row in combined)
    for condition in conditions:
        print(f"{condition}: {counts[condition]} records")

    print("Original input files were not modified.")


if __name__ == "__main__":
    main()