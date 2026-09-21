"""
STEP 2: Run the full experiment.

For each sentence in dataset.csv, and for each of the 3 prompting conditions
(zero-shot, few-shot, chain-of-thought), this script calls the local LLM
and records the result. Sentences in the "instability subset" get 2 extra
repeat runs per condition to measure output consistency.

IMPORTANT - RESUMABLE DESIGN:
Every single result is appended to results_llm.csv immediately after it's
received. If the script crashes, hangs, or you stop it (Ctrl+C), you can
just run it again -- it will skip everything already completed and continue
from where it left off.

Estimated total runtime: roughly 70-90 minutes depending on your machine
(see the printed estimate at startup).

Run order:
    1. python 1_data_prep.py        (creates dataset.csv)
    2. python 2_run_experiment.py   (this script - the long one)
    3. python 3_run_vader.py        (fast, seconds)
    4. python 4_analyze_results.py  (produces tables + chart)
"""

import csv
import os
import time
from prompts import zero_shot_prompt, few_shot_prompt, cot_prompt, parse_final_label
from llm_client import call_llm

DATASET_FILE = "dataset.csv"
SUBSET_FILE = "instability_subset_ids.csv"
RESULTS_FILE = "results_few_shot_v2.csv"

CONDITIONS = {
    "few_shot": few_shot_prompt,
}

RESULT_FIELDS = ["id", "text", "gold_label", "category", "condition", "run_number", "raw_response", "parsed_label"]


def load_dataset():
    with open(DATASET_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_instability_subset():
    with open(SUBSET_FILE, newline="", encoding="utf-8") as f:
        return set(int(row["id"]) for row in csv.DictReader(f))


def load_completed_keys():
    """Returns a set of (id, condition, run_number) tuples already done."""
    completed = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                completed.add((int(row["id"]), row["condition"], int(row["run_number"])))
    return completed


def append_result(row: dict):
    file_exists = os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()  # ensure it's written to disk immediately


def build_task_list(dataset, instability_ids):
    """
    Builds the full list of (row, condition_name, run_number) tasks.
    Run 1 happens for every sentence in every condition.
    Runs 2 and 3 happen only for sentences in the instability subset.
    """
    tasks = []
    for row in dataset:
        row_id = int(row["id"])
        for condition_name in CONDITIONS:
            tasks.append((row, condition_name, 1))
            if row_id in instability_ids:
                tasks.append((row, condition_name, 2))
                tasks.append((row, condition_name, 3))
    return tasks


def main():
    dataset = load_dataset()
    instability_ids = load_instability_subset()
    completed = load_completed_keys()

    all_tasks = build_task_list(dataset, instability_ids)
    remaining_tasks = [
        t for t in all_tasks
        if (int(t[0]["id"]), t[1], t[2]) not in completed
    ]

    print(f"Total tasks: {len(all_tasks)}")
    print(f"Already completed (resuming): {len(all_tasks) - len(remaining_tasks)}")
    print(f"Remaining: {len(remaining_tasks)}")
    print(f"Estimated time remaining: ~{len(remaining_tasks) * 9 / 60:.0f} minutes (at ~9s/call)\n")

    if not remaining_tasks:
        print("Nothing left to do -- all results already collected!")
        return

    start_time = time.time()
    for i, (row, condition_name, run_number) in enumerate(remaining_tasks):
        prompt_fn = CONDITIONS[condition_name]
        prompt = prompt_fn(row["text"])

        # Temperature is fixed at 0.7 for ALL runs (not just repeats) so that
        # the single "main" run and the repeat runs are directly comparable,
        # and so observed instability reflects realistic LLM usage rather
        # than an artificially deterministic setting.
        try:
            raw_response = call_llm(prompt, temperature=0.7, timeout=60, max_retries=2)
        except RuntimeError as e:
            print(f"SKIPPING id={row['id']} condition={condition_name} run={run_number} due to error: {e}")
            continue

        parsed_label = parse_final_label(raw_response)

        result_row = {
            "id": row["id"],
            "text": row["text"],
            "gold_label": row["gold_label"],
            "category": row["category"],
            "condition": condition_name,
            "run_number": run_number,
            "raw_response": raw_response.replace("\n", " | "),  # keep CSV single-line
            "parsed_label": parsed_label,
        }
        append_result(result_row)

        elapsed = time.time() - start_time
        avg_per_task = elapsed / (i + 1)
        remaining = len(remaining_tasks) - (i + 1)
        eta_min = remaining * avg_per_task / 60

        print(
            f"[{i + 1}/{len(remaining_tasks)}] id={row['id']} {condition_name} run={run_number} "
            f"-> {parsed_label} (gold={row['gold_label']}) | ETA: {eta_min:.1f} min remaining"
        )

    print("\nDone! All results saved to results_llm.csv")


if __name__ == "__main__":
    main()
