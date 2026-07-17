# Dataiku OpenLayer Pipeline

This folder contains the files and instructions to run an LLM evaluation flow in Dataiku with OpenLayer.
These instructions use one standard starter schema in Dataiku. The benchmark results that are retrieved depend on the benchmark/test suite configured in the target OpenLayer project.
Use this README as the single setup and handoff document for the Dataiku flow.

Recommended high-level flow:

1. Request the internal access you need: GitHub, Vault, and the target OpenLayer project.
2. Create an OpenLayer account and request access to the correct OpenLayer project.
3. In OpenLayer, confirm the target project already has the intended benchmark/test suite and configure the LLM provider API variables needed for LLM-as-a-judge benchmarks.
4. Import the client file into Dataiku.
5. Build a prepared dataset with a Prepare recipe.
6. Create a Python recipe, first copy and paste the evaluation script, and then modify the required code settings.
7. Run the recipe so each execution appends one benchmark row.
8. Create a Sync recipe so the history is available as a resulting dataset for analysis.

Before you start the setup, make sure you already know these project-specific values:

- Which OpenLayer project should receive the runs
- The OPENLAYER_API_KEY value
- The OPENLAYER_PROJECT_ID value
- Which source columns should map to input, context, ground_truth, and generated_summary
- The Dataiku dataset names you want to use for the prepared dataset and benchmark history dataset

## 1. Before You Start

Before configuring Dataiku, make sure the user has completed the access and setup steps below:

1. Request both GitHub and OpenLayer access so you can access the repository and the required OpenLayer workspace.
2. Sign in to GitHub and the enterprise OpenLayer workspace using your company single sign-on (SSO).
3. If applicable, ensure your team's password manager or vault is set up to access required credentials.
4. Create an OpenLayer account at https://app.openlayer.com using your company credentials when creating the account.
5. Use the OpenLayer SSO login page at https://app.openlayer.com/sso when signing in to OpenLayer.
6. Request access to the OpenLayer project that should receive the benchmark runs.
7. Confirm that this OpenLayer project already has the intended benchmark/test suite configured for the evaluation you want to run.
8. Request access to the Vault that stores the OpenLayer LLM credentials used for LLM-as-a-judge benchmarks.
9. Obtain the OpenLayer API key.
10. Obtain the OpenLayer project id.
11. In OpenLayer, open Settings > Environment variables and configure the LLM provider API variables required for LLM-as-a-judge benchmarks.
12. Make sure you can open this repository and copy the Python file from this folder into the Dataiku recipe editor.

For the OpenLayer LLM provider variables, use the variable names required by the provider configuration already used in your OpenLayer workspace. If you do not know which ones are required, ask the OpenLayer project owner before running the recipe.

For the documented copy-paste flow, you first copy the Python recipe file from this repository into Dataiku and then modify the required settings directly in the Dataiku recipe editor.

The Python recipe will not run successfully without OpenLayer project access.

OpenLayer benchmark documentation reference:

- Tests overview: https://docs.openlayer.com/tests/overview.md
- Browse the benchmark catalog: https://docs.openlayer.com/tests/browse.md
- Tests configuration: https://docs.openlayer.com/tests/test-configuration.md

## 2. Set Dataiku Project Variables

Set these variables in Dataiku Project Variables.

These values should be configured in the Dataiku project itself before running the recipe.
Do not rely only on local shell environment variables on your own machine. The Dataiku project, recipes, and scenarios must be able to read them from the project context.

For the basic copy-paste Python benchmark recipe, the only variables you need to start are:

- OPENLAYER_API_KEY
- OPENLAYER_PROJECT_ID

You can ignore the other optional variables unless your team specifically needs them.

Required:

- OPENLAYER_API_KEY
- OPENLAYER_PROJECT_ID

Optional:

- OPENLAYER_API_URL
- OPENLAYER_METADATA_ENV
- OPENLAYER_FOLDER_ID (only for managed-folder upload or managed-folder log export flows; not required by dataiku_evaluation_recipe.py)

These variables are optional because the recipe can run with the default OpenLayer API endpoint and can infer metadata defaults when no overrides are provided.

- OPENLAYER_API_URL is only needed when your OpenLayer workspace uses a non-default API endpoint.
  If it is not set, the recipe uses the standard OpenLayer API URL.
- OPENLAYER_METADATA_ENV is used to pass metadata defaults such as dataset name, dataset version, and LLM name without editing the Python code.
  If it is not set, the recipe falls back to values inferred from the input dataset, the current run date, and existing columns.
- OPENLAYER_FOLDER_ID is only used when you upload outputs to a Dataiku managed folder or build follow-up datasets from that managed folder.
  It is not used by the simple copy-paste Python benchmark recipe.

How to get these values:

- OPENLAYER_API_KEY
  Create it in OpenLayer from the user menu by opening API Keys and creating a new key.
  Store it securely and set it in the Dataiku project.
- OPENLAYER_PROJECT_ID
  Open the target project in OpenLayer and copy the id shown in the project details/settings page or the id that appears in the project URL after `/projects/`.
  If you still cannot identify it, ask the project owner or reuse the id from an existing working team configuration.
- OPENLAYER_API_URL
  For the default OpenLayer cloud, use `https://api.openlayer.com/v1`.
  Only override it if your workspace uses a different API base URL.
- OPENLAYER_FOLDER_ID
  Use the id of the Dataiku managed folder only if your flow uploads outputs to a managed folder or reads logs back from one.
  Get this value from the target managed folder in Dataiku.

Use OPENLAYER_METADATA_ENV when you want to control the dataset and model naming without editing the Python code.

Example format for the value of OPENLAYER_METADATA_ENV:

```text
OPENLAYER_LLM_NAME=gpt-4.1-mini
OPENLAYER_DATASET_VERSION=llm_evaluation_v1
```

Do not use JSON for this variable in the current copy-paste recipe.

## 3. Create the Prepared Dataset With a Prepare Recipe

The source dataset in Dataiku may contain many columns, but the prepared dataset for OpenLayer should expose a small set of exact column names.
This starter recipe expects the exact column names below, regardless of the task-specific source schema.

Create a Prepare recipe from the imported source dataset.

The prepared dataset should contain these exact column names:

- **`input`:**
  This is the main source input for the model task.
  OpenLayer uses this as the primary task input, so this column should contain the real source content and not a shortened note or metadata field.
- **`context`:**
  This is the context that should be available during evaluation.
  If your process uses a narrower or cleaner evidence field than the full input, place that content here instead.
  The important point is that context should contain the information the evaluator can use to judge whether the model output is grounded.
- **`ground_truth`:**
  This is the reference answer used for evaluation.
  Benchmarks that compare the model result against a trusted reference rely on this column, so it should contain the best available target output for that row.
- **`generated_summary`:**
  This is the model-generated output you want to evaluate.
  The starter recipe keeps this exact column name as part of the standard schema, even when the model output is not literally a summary.
  There should be one generated output per row, aligned with the corresponding input, context, and ground_truth.
- **`output`:**
  This should be an exact copy of generated_summary.
  The current OpenLayer runtime expects an output column with this name, so even if generated_summary already exists, output must also be present.
  In this flow, output and generated_summary should contain the same value for every row.

Task-specific mapping examples:

- **Summary evaluation:**
  - `input`: the original transcript, document, or source text
  - `context`: usually the same as `input`, or a cleaner evidence field
  - `ground_truth`: the trusted human-written or approved summary
  - `generated_summary`: the model-generated summary
- **RAG or grounded-answer evaluation:**
  - `input`: the user query, question, or task prompt
  - `context`: the retrieved passages or documents that were actually supplied to the model for that row
  - `ground_truth`: a trusted reference answer if you have one; this is not the retrieved context
  - `generated_summary`: the model-generated answer, even if the column name still says summary

For RAG, if more than one document was retrieved, join the retrieved evidence into the single `context` field for that row. Do not place the full source corpus in `context`; use only the material that the model actually received for that answer.

Current limitation of the starter recipe:

- The current copy-paste Python recipe still enforces non-empty `ground_truth` values and the shipped `rag` profile also points to `ground_truth` as the reference column.
- If your current RAG flow only has query + retrieved context + generated answer, but no trusted reference answer, this documented starter flow is not a direct fit yet. In that case you would need a small recipe/profile change before using it as a pure grounded-answer check.

Recommended optional columns:

- **`conversation_id`:**
  A stable row or conversation identifier used to trace a benchmark result back to the original source record.
  This is useful for debugging, filtering, and joining results back to the original dataset.
- **`model_name`:**
  The name of the model or system that produced generated_summary.
  This is useful when comparing runs across multiple model versions or when storing benchmark history over time.
- **`prompt_version`:**
  The prompt or template version used to produce the row output.
  Keep this column when you want benchmark history to stay traceable across prompt revisions.
- **`prompt_parameters_json`:**
  A JSON object serialized as text containing the prompt parameters used for that row.
  Use this when the evaluation should retain prompt configuration details such as temperature, template switches, or retrieval settings.

Any user can map their own source columns to these names in the Prepare recipe. The source column names do not matter, as long as the prepared dataset ends with the exact OpenLayer-ready names above.

Suggested Prepare recipe steps:

1. Map the source task input column to input.
2. Map the evidence or context column to context, or copy input into context when that is appropriate for your flow. For RAG, use the retrieved passages or documents that were actually passed to the model.
3. Map the trusted reference output column to ground_truth. For RAG, this should be a reference answer if one exists, not the retrieved passages.
4. Map the model-generated output column to generated_summary. For RAG, this is the generated answer.
5. Create output as a copy of generated_summary.
6. Optionally keep conversation_id and model_name for traceability.
7. If available, keep prompt_version and prompt_parameters_json in the prepared dataset.
8. Trim input, context, ground_truth, generated_summary, and output.
9. Remove rows where required fields are empty.
10. Truncate input to 49000 characters.
11. Truncate generated_summary and output to 2000 characters.
12. Filter out rows where generated_summary is shorter than 20 characters.
13. Remove duplicates on input, ground_truth, and generated_summary.

Example prepared dataset name:

- prepared_openlayer_evaluation

## 4. Create the Python Recipe

After the prepared dataset has been built, create a Python recipe in Dataiku.

Recommended setup:

1. Input dataset: the prepared dataset from the Prepare recipe.
2. Output dataset: a benchmark history dataset, for example benchmarks_output1.
3. In Dataiku, select an established code environment with the required dependencies for this Python recipe when it is available.
  This is the preferred option to get the script running faster because the required packages are already preinstalled there.
4. In the Dataiku recipe output settings, set the output write mode to Append.

If a pre-configured code environment does not exist in your Dataiku instance, ask the Dataiku administrator or the team owning this flow which code environment should be used instead.

Open the Python recipe editor. For this documented flow, first copy and paste the full contents of:

- scalable-oversight-dataiku-adapter/dataiku_evaluation_recipe.py

Then review and update these values in the code:

1. INPUT_DATASET_NAME
   Set this to the name of the prepared dataset created in Dataiku.
2. OUTPUT_DATASET_NAME
   Set this to the benchmark history dataset that should receive the results.
3. MAX_ROWS
  The starter recipe evaluates the first 101 valid prepared rows per run because MAX_ROWS is set to 101 near the top of the script.
  Change MAX_ROWS there if you want to use a different row limit.

Optional naming changes:

1. LLM version name
   This can be controlled through OPENLAYER_METADATA_ENV with OPENLAYER_LLM_NAME.
2. Database version output value
  The recipe writes this value into the `database_version` output column.
  It can be controlled through OPENLAYER_METADATA_ENV with OPENLAYER_DATASET_VERSION.
3. Prompt version output value
  The recipe writes this value into the `prompt_version` output column.
  It can be controlled through OPENLAYER_METADATA_ENV with OPENLAYER_PROMPT_VERSION.

Prompt parameter handling:

1. If the prepared dataset includes `prompt_parameters_json`, the recipe keeps that column in the OpenLayer input table.
2. When all evaluated rows share the same JSON object, the recipe also forwards it into OpenLayer run metadata parameters.
3. If prompt parameters differ across rows in one run, the recipe keeps the row-level input column and writes `mixed` at the aggregated run level.

If those metadata values are not set, the script falls back to inferred defaults.

## 5. Run the Python Recipe

Run the Python recipe after the variables and dataset names are configured.

Important:

- The output dataset should be configured in Append mode in Dataiku.
- If Append mode is not enabled, each run can overwrite the previous benchmark history.
- Use an OpenLayer project that already has the intended benchmark/test suite configured before the first run.
- The benchmark columns written to the output dataset come from the benchmarks/tests configured in that OpenLayer project.

Each successful run should append one aggregated run-level row to the output dataset.
The recipe evaluates multiple prepared dataset rows in one run and then writes one history row with the timestamp, version fields, and benchmark results for that run.
In the starter version of the script, this run evaluates only the first 101 valid prepared rows unless you change MAX_ROWS in the code.

The output contains:

- timestamp_utc
- database_version
- llm_version
- prompt_version
- prompt_parameters_json
- one numeric column per benchmark

This dataset becomes the run history for the evaluation flow.

First-run validation:

1. Confirm that the Dataiku Python recipe run finishes successfully.
2. Confirm that exactly one new row is appended to the benchmark history dataset.
3. Confirm that timestamp_utc, database_version, llm_version, and prompt_version are populated in that new row.
4. Confirm that prompt_parameters_json is populated when the prepared dataset includes a stable prompt parameter object for the run.
5. Confirm that the benchmark columns contain numeric values for the run.
6. Confirm in OpenLayer that the run appears in the expected project and that the intended benchmarks/tests were used.

## 6. Create a Sync Recipe for the Resulting Dataset

After the Python recipe is working and the first-run validation checks pass, create a Sync recipe in Dataiku so the benchmark history is available as the resulting dataset used for charts, dashboards, and downstream analysis.

The Python recipe output already contains the benchmark history. The Sync recipe is recommended as a clean publication step so you keep the recipe output as the technical working dataset and expose a separate final dataset for analysis, dashboards, and downstream users.

Recommended pattern:

1. Use the Python recipe output dataset as the source history dataset.
2. Create a Sync recipe from that history dataset to the final resulting dataset used for analysis.
3. Use that resulting dataset in Charts, Statistics, Dashboards, and Scenarios.

If all runs already append into a single benchmark history dataset, the Sync recipe is the final publication step.

If in the future the logs are split across multiple intermediate datasets, add a Stack recipe before the Sync recipe so all logs are combined into one history dataset first.

The final resulting dataset can then be analyzed directly in Dataiku.

How to analyze it in the Dataiku Statistics section:

1. Open the final resulting dataset in Dataiku.
2. Go to the Statistics tab.
3. Refresh or compute statistics if Dataiku asks for it.
4. Review the benchmark columns to see summary statistics such as min, max, mean, and value distribution.
5. Review `timestamp_utc`, `database_version`, and `llm_version` to understand which runs and model versions are present in the dataset.
6. Use the column-level statistics views to spot missing values, unusual benchmark values, and changes across runs.

For trend analysis or model comparisons, the same final dataset can also be opened in the Charts section.

## 7. Quick Troubleshooting

- If the recipe cannot connect to OpenLayer, recheck OPENLAYER_API_KEY and OPENLAYER_PROJECT_ID in Dataiku Project Variables.
- If LLM-as-a-judge benchmarks fail, recheck the provider environment variables in OpenLayer Settings > Environment variables.
- If runs overwrite older results, recheck that the output dataset is set to Append mode.
- If the recipe cannot start in Dataiku, recheck that the selected code environment is correct and available.

## 8. Files in This Folder

- dataiku_evaluation_recipe.py
  Copy-paste Dataiku Python recipe for benchmark extraction in the starter evaluation flow.
- dataiku_scenario_runner.py
  Scenario-oriented runner for a broader automated flow.
- dataiku_audit_export_builder.py
  Helper script for building Dataiku datasets from exported log artifacts.

## 8. Recommended End State in Dataiku

The recommended Dataiku flow for this evaluation use case is:

1. Source client dataset
2. Prepare recipe
3. Prepared evaluation dataset
4. Python recipe with OpenLayer evaluation
5. Benchmark history dataset
6. Sync recipe
7. Final resulting dataset for charts and analysis
