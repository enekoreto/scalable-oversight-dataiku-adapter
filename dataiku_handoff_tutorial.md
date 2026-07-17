# Dataiku + OpenLayer Client Tutorial (Summaries Use Case)

This tutorial explains how to run the summaries evaluation pipeline in Dataiku, including:

- which environment variables are required
- which dataset columns must exist
- how to prepare the input dataset in Dataiku
- how to run the Python recipe and validate outputs

Scope: this guide is for the summaries flow. A future RAG extension can reuse the same structure with additional fields.

## 1. Prerequisites

Before starting, complete these steps:

1. Request GitHub access in KPN IAM so you can access the repository and related project files.
2. Make sure you have a Dataiku project with permission to create recipes and datasets.
3. Create an OpenLayer account at https://app.openlayer.com using KPN credentials.
4. Request access to the correct OpenLayer project.
5. Obtain the OpenLayer API key and OpenLayer project id.
6. In OpenLayer, open Settings > Environment variables and configure the LLM provider API variables required for LLM-as-a-judge benchmarks.
7. Make sure the repository files are available to the Dataiku runtime.

OpenLayer benchmark documentation reference:

- Tests overview: https://docs.openlayer.com/tests/overview.md
- Browse the benchmark catalog: https://docs.openlayer.com/tests/browse.md
- Tests configuration: https://docs.openlayer.com/tests/test-configuration.md

## 2. Required Dataiku Variables

Set these variables in Dataiku Project Variables.

These values should be configured in the Dataiku project itself before running the recipe.
Do not rely only on local shell environment variables on your own machine. The Dataiku project, recipes, and scenarios must be able to read them from the project context.
If your Dataiku setup also uses a local execution environment for the recipe, mirror the same values there as environment variables.

Required:

- `OPENLAYER_API_KEY`
- `OPENLAYER_PROJECT_ID`

Optional:

- `OPENLAYER_API_URL` (only if you use a non-default OpenLayer API endpoint)
- `OPENLAYER_METADATA_ENV` (optional multiline env-like text for metadata defaults)
- `OPENLAYER_FOLDER_ID` (only for managed-folder upload or managed-folder log export flows; not required by `dataiku_evaluation_recipe.py`)

These variables are optional because the recipe can run with the default OpenLayer API endpoint and can infer metadata defaults when no overrides are provided.

- `OPENLAYER_API_URL` is used only when your OpenLayer workspace uses a non-default API endpoint.
  If it is not set, the recipe uses the standard OpenLayer API URL.
- `OPENLAYER_METADATA_ENV` is used to provide metadata defaults such as dataset name, dataset version, and LLM name without editing the Python recipe.
  If it is not set, the recipe falls back to values inferred from the input dataset, the current run date, and existing columns.
- `OPENLAYER_FOLDER_ID` is only used when you upload outputs to a Dataiku managed folder or build follow-up datasets from that managed folder.
  It is not used by the simple copy-paste Python benchmark recipe.

How to get these values:

- `OPENLAYER_API_KEY`
  Create it in OpenLayer from the user menu by opening API Keys and creating a new key.
  Store it securely and set it in the Dataiku project or local Dataiku execution environment.
- `OPENLAYER_PROJECT_ID`
  Use the id of the target OpenLayer project that should receive the benchmark runs.
  If you do not know the correct project id, ask the project owner or reuse the id from an existing working team configuration.
- `OPENLAYER_API_URL`
  For the default OpenLayer cloud, use `https://api.openlayer.com/v1`.
  Only override it if your workspace uses a different API base URL.
- `OPENLAYER_FOLDER_ID`
  Use the id of the Dataiku managed folder only if your flow uploads outputs to a managed folder or reads logs back from one.
  Get this value from the target managed folder in Dataiku.

## 3. Required Input Dataset Columns (Summaries)

Your prepared dataset must contain these columns:

- **`input`:**
  This is the original source content that the model had to summarize.
  For summary evaluations, it is usually the transcript, conversation, document, note, or source text from which the summary was produced.
  OpenLayer uses this as the main evidence behind the task, so this column should contain the real underlying text and not a shortened note or metadata field.
- **`context`:**
  This is the context that should be available during evaluation.
  In many summary workflows, context is identical to `input`, so copying `input` into `context` is correct.
  If your process uses a narrower or cleaner evidence field than the full input, place that content here instead.
- **`ground_truth`:**
  This is the reference answer used for evaluation.
  For summary evaluations, it should be the human-written, reviewed, or otherwise approved summary that represents the expected correct output.
  Benchmarks that compare the model result against a trusted reference rely on this column.
- **`generated_summary`:**
  This is the summary produced by the model version you want to evaluate.
  There should be one generated summary per row, aligned with the corresponding `input`, `context`, and `ground_truth`.
  If you want to evaluate the raw model output, this column should contain the exact summary returned by the model rather than a manually edited version.
- **`output`:**
  This should be an exact copy of `generated_summary`.
  The current OpenLayer runtime expects an `output` column with this name, so even if `generated_summary` already exists, `output` must also be present.
  In this flow, `output` and `generated_summary` should contain the same value for every row.

Recommended extra columns:

- **`conversation_id`:**
  A stable row or conversation identifier used to trace a benchmark result back to the original source record.
  This is useful for debugging, filtering, and joining results back to the original dataset.
- **`model_name`:**
  The name of the model or system that produced `generated_summary`.
  This is useful when comparing runs across multiple model versions or when storing benchmark history over time.
- **`prompt_version`:**
  The prompt or template version used to produce the generated summary for that row.
  Keep this when you want downstream analysis to compare benchmark results across prompt revisions.
- **`prompt_parameters_json`:**
  A JSON object serialized as text containing the prompt parameters for that row.
  Use this when prompt settings such as temperature, template switches, or retrieval settings should stay visible in run history.

## 4. Data Quality Rules (for current OpenLayer tests)

To stay compatible with existing tests and limits:

- `input`, `ground_truth`, `generated_summary`, `output` must be non-empty
- trim whitespace on text columns
- keep `generated_summary` length between 20 and 2000 characters
- keep `input` length <= 49000 characters
- remove duplicates on at least (`input`, `ground_truth`, `generated_summary`)
- keep at least 10 rows (preferred) and at most 100000 rows

## 5. Build the Prepared Dataset in Dataiku (Step by Step)

### Step 5.1: Create source dataset

1. Import the client file (for example Excel) into Dataiku.
2. Confirm the source dataset schema is detected correctly.

### Step 5.2: Create a Prepare recipe

1. Create a Prepare recipe from the source dataset.
2. Create output dataset (example): `MI_filtered_Ground_truth__version_2__Transcripties_prepared`.

### Step 5.3: Transform columns

In the Prepare recipe, add these transformations:

1. Map the source text column to `input`.
2. Map a separate context column to `context`, or copy `input` into `context`.
3. Map the reference answer or reference summary column to `ground_truth`.
4. Map the model output or generated summary column to `generated_summary`.
5. Create `output` as a copy of `generated_summary`.
6. (Optional) Keep or create `conversation_id` and `model_name` if you want traceability in downstream analysis.
7. (Optional but recommended) Keep `prompt_version` and `prompt_parameters_json` when you want prompt-level traceability.

### Step 5.4: Clean and validate

Add prepare steps:

1. Trim `input`, `context`, `ground_truth`, `generated_summary`, `output`.
2. Remove rows where required fields are empty.
3. Truncate `input` to 49000 chars.
4. Truncate `generated_summary` (and `output`) to 2000 chars.
5. Filter rows where `generated_summary` length < 20.
6. Remove duplicates on (`input`, `ground_truth`, `generated_summary`).

Build the prepared dataset.

## 6. Configure and Run the Python Recipe

Use the OpenLayer Python recipe with:

- input dataset: prepared dataset from Step 5
- output dataset: results history dataset (example: `benchmarks_output1`)
- selected Dataiku code environment: `starni_evaluation_pipeline`

In the script, make sure constants match:

- `INPUT_DATASET_NAME = "MI_filtered_Ground_truth__version_2__Transcripties_prepared"`
- `OUTPUT_DATASET_NAME = "benchmarks_output1"`

Run the recipe.

## 7. Expected Output Behavior

Each run appends one new row in the output dataset (history behavior).

The output row contains:

- run metadata (timestamp, dataset/model info, prompt info, project version id, status)
- benchmark numeric result columns (flattened from OpenLayer observed values)
- benchmark thresholds where available
- diagnostics columns (`push_stdout`, `push_stderr`, `inspect_status`, etc.)

The resulting dataset can be analyzed directly in Dataiku.

How to analyze it in the Statistics section:

1. Open the resulting dataset in Dataiku.
2. Go to the Statistics tab.
3. Refresh or compute statistics if Dataiku asks for it.
4. Review the benchmark columns to inspect summary statistics such as min, max, mean, and value distribution.
5. Review `timestamp_utc`, `database_version`, `llm_version`, and `prompt_version` to understand which runs, model versions, and prompt variants are included.
6. Use the column-level statistics views to spot missing values, unusual benchmark values, and changes across runs.

If you want visual comparisons over time or across model versions, use the same dataset in the Charts section as well.

## 8. Post-Run Validation Checklist

After each run, verify:

- output row count increased by 1
- `status` is `success` (or `failed` with diagnostics)
- `project_version_id` is populated
- `prompt_version` is populated when provided through the prepared dataset or Dataiku metadata variables
- `benchmark_result_count` > 0 when benchmark retrieval succeeds
- benchmark numeric columns are present (columns starting with `benchmark_`)

## 9. Troubleshooting

If run fails, check these first:

- `OPENLAYER_API_KEY` and `OPENLAYER_PROJECT_ID` are set
- required input columns exist with exact names
- prepared dataset does not contain empty required text fields
- input text is not too long (`input` > 49000 chars)
- output summary is not too short/long (outside 20..2000 chars)

Use diagnostic fields in output dataset:

- `push_stderr`
- `inspect_stderr`
- `benchmark_fetch_error`

## 10. Future Extension to RAG

When extending this to RAG, keep the same pipeline shape:

- same run metadata pattern
- same append-per-run output behavior
- same environment variable strategy

Expected additions for RAG are typically dataset schema and test profile changes (for example retrieval-specific context fields and RAG metrics), while operational setup remains the same.
