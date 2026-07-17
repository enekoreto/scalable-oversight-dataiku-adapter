# -*- coding: utf-8 -*-
"""Dataiku helper recipe: build log_out from managed-folder CSV artifacts.

Usage in a Dataiku Python recipe:
1. Add the managed folder as recipe input.
2. Add output dataset named in OUTPUT_DATASET.
3. Paste and run this script.
"""

import dataiku
import pandas as pd

MANAGED_FOLDER_ID = "YOUR_MANAGED_FOLDER_ID"  # Replace with your actual managed folder ID
OUTPUT_DATASET = "log_out"

folder = dataiku.Folder(MANAGED_FOLDER_ID)
paths = folder.list_paths_in_partition()
print("Managed folder files:", paths)


def read_csv_from_folder(candidate_filenames):
    for filename in candidate_filenames:
        candidates = [p for p in paths if p == filename or p.endswith("/" + filename)]
        if not candidates:
            continue

        chosen = candidates[0]
        print("Using", chosen, "for", filename)
        with folder.get_download_stream(chosen) as stream:
            return pd.read_csv(stream)

    raise Exception(
        "Missing any candidate file in managed folder: " + str(candidate_filenames) + " | Found: " + str(paths)
    )


RUNS_CANDIDATES = [
    "openlayer_runs_history.csv",
    "openlayer_runs.csv",
    "openlayer_runs_latest.csv",
]

BENCHMARKS_CANDIDATES = [
    "openlayer_benchmark_thresholds_history.csv",
    "openlayer_benchmark_thresholds.csv",
    "openlayer_benchmark_thresholds_latest.csv",
]

runs_df = read_csv_from_folder(RUNS_CANDIDATES)
runs_df["source_table"] = "openlayer_runs"

bench_df = read_csv_from_folder(BENCHMARKS_CANDIDATES)
bench_df["source_table"] = "openlayer_benchmark_thresholds"

log_out_df = pd.concat([runs_df, bench_df], ignore_index=True, sort=False)

log_out = dataiku.Dataset(OUTPUT_DATASET)
log_out.write_with_schema(log_out_df)

print("Rows written to log_out:", len(log_out_df))
print("Runs rows:", len(runs_df))
print("Benchmark rows:", len(bench_df))
