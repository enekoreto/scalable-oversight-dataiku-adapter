# -*- coding: utf-8 -*-
import os
import re
import sys
import json
import time
import shutil
import tarfile
import platform
import importlib
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlparse, urlunparse

import dataiku
import pandas as pd

INPUT_DATASET_NAME = "GPT_4_V1_prepared"
OUTPUT_DATASET_NAME = "benchmarks_output1"

MAX_ROWS = 101
MAX_INPUT_CHARS = 49000
MAX_SUMMARY_CHARS = 2000
MIN_SUMMARY_CHARS = 20
DEFAULT_PROMPT_VERSION = "unknown"
DEFAULT_PROMPT_PARAMETERS_JSON = "{}"

IGNORED_OBSERVED_KEYS = {
    "llmevaluator",
    "erroredmetrics",
    "skippedmetrics",
}


def parse_env_text(text):
    parsed = {}
    if not text:
        return parsed

    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]

        if key:
            parsed[key] = value

    return parsed


def trim_text(value, max_len=12000):
    text = str(value or "")
    if len(text) <= max_len:
        return text
    return text[:max_len] + " ...[truncated]"


def strip_ansi(text):
    return re.sub(r"\x1b\[[0-9;]*m", "", str(text or ""))


def extract_openlayer_status(text):
    cleaned = strip_ansi(text).lower()
    for line in cleaned.splitlines():
        line = line.strip()
        if line.startswith("status:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def extract_project_version_id(text):
    cleaned = strip_ansi(text)

    match = re.search(r"projectVersionId:\s*([0-9a-fA-F-]{36})", cleaned, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r"Version ID:\s*([0-9a-fA-F-]{36})", cleaned, flags=re.IGNORECASE)
    if match:
        return match.group(1)

    return ""


def sanitize_key(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "unknown"


def infer_model_name(df):
    candidate_cols = [
        "model_name",
        "model",
        "llm_name",
        "model_used",
        "generator_model",
        "model_id",
    ]

    for col in candidate_cols:
        if col in df.columns:
            values = [str(v).strip() for v in df[col].dropna().astype(str).tolist() if str(v).strip()]
            unique_vals = sorted(set(values))
            if len(unique_vals) == 1:
                return unique_vals[0]
            if len(unique_vals) > 1:
                preview = ",".join(unique_vals[:3])
                if len(unique_vals) > 3:
                    preview += ",..."
                return "multi:" + preview

    summary_model_cols = sorted([c for c in df.columns if c.startswith("summary_model_")])
    if summary_model_cols:
        labels = [c.replace("summary_", "") for c in summary_model_cols]
        if len(labels) == 1:
            return labels[0]
        return "multi:" + ",".join(labels)

    return "unknown-model"


def summarize_text_values(series):
    values = [str(v).strip() for v in series.fillna("").astype(str).tolist() if str(v).strip()]
    unique_vals = sorted(set(values))
    if not unique_vals:
        return DEFAULT_PROMPT_VERSION
    if len(unique_vals) == 1:
        return unique_vals[0]
    return "mixed"


def normalize_json_object_text(value):
    text = str(value or "").strip()
    if not text:
        return "", None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text, None

    if not isinstance(parsed, dict):
        return text, None

    normalized = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return normalized, parsed


def summarize_prompt_parameters(series):
    normalized_values = []
    parsed_by_value = {}

    for raw_value in series.fillna("").astype(str).tolist():
        normalized_value, parsed_value = normalize_json_object_text(raw_value)
        if not normalized_value:
            continue
        normalized_values.append(normalized_value)
        if parsed_value is not None:
            parsed_by_value[normalized_value] = parsed_value

    unique_vals = sorted(set(normalized_values))
    if not unique_vals:
        return DEFAULT_PROMPT_PARAMETERS_JSON, {}
    if len(unique_vals) > 1:
        return "mixed", None

    single_value = unique_vals[0]
    return single_value, parsed_by_value.get(single_value)


def first_existing_column(columns, candidates):
    lookup = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        hit = lookup.get(candidate.lower())
        if hit is not None:
            return hit
    return None


def coerce_series(df, column_name):
    if column_name is None:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[column_name].fillna("").astype(str)


def prepare_openlayer_runtime_files(input_df, runtime_dir, dataset_name):
    cols = list(input_df.columns)

    output_col = first_existing_column(
        cols,
        ["generated_summary", "output", "prediction", "model_output", "answer", "summary"],
    )
    input_col = first_existing_column(
        cols,
        ["input", "prompt", "question", "source", "document", "source_text"],
    )
    context_col = first_existing_column(
        cols,
        ["context", "article", "passage", "background", "source_text"],
    )
    ground_truth_col = first_existing_column(
        cols,
        ["ground_truth", "reference_summary", "reference", "target", "expected_output"],
    )
    prompt_version_col = first_existing_column(
        cols,
        ["prompt_version", "promptVersion", "template_version", "prompt_template_version"],
    )
    prompt_parameters_col = first_existing_column(
        cols,
        ["prompt_parameters_json", "prompt_parameters", "parameters_json", "prompt_params_json"],
    )

    if output_col is None:
        output_col = cols[-1] if cols else None
    if output_col is None:
        raise Exception("Input dataset has no columns, cannot build OpenLayer config")

    if input_col is None:
        for col in cols:
            if col != output_col:
                input_col = col
                break
    if input_col is None:
        input_col = output_col

    prepared = pd.DataFrame()
    prepared["input"] = coerce_series(input_df, input_col).astype(str).str.strip().str.slice(0, MAX_INPUT_CHARS)

    if context_col:
        prepared["context"] = coerce_series(input_df, context_col).astype(str).str.strip()
    else:
        prepared["context"] = prepared["input"]

    if ground_truth_col:
        prepared["ground_truth"] = coerce_series(input_df, ground_truth_col).astype(str).str.strip()
    else:
        prepared["ground_truth"] = ""

    prepared["generated_summary"] = coerce_series(input_df, output_col).astype(str).str.strip().str.slice(0, MAX_SUMMARY_CHARS)
    prepared["output"] = prepared["generated_summary"]

    if prompt_version_col:
        prepared["prompt_version"] = coerce_series(input_df, prompt_version_col).astype(str).str.strip()

    if prompt_parameters_col:
        prepared["prompt_parameters_json"] = coerce_series(input_df, prompt_parameters_col).astype(str).str.strip()

    prepared = prepared[
        (prepared["input"].str.len() > 0)
        & (prepared["context"].str.len() > 0)
        & (prepared["ground_truth"].str.len() > 0)
        & (prepared["generated_summary"].str.len() >= MIN_SUMMARY_CHARS)
    ].copy()

    if prepared.empty:
        raise Exception("No valid rows after enforcing required columns and length constraints.")

    prepared = prepared.head(MAX_ROWS).copy()

    input_variable_names = ["input"]
    if "prompt_version" in prepared.columns:
        input_variable_names.append("prompt_version")
    if "prompt_parameters_json" in prepared.columns:
        input_variable_names.append("prompt_parameters_json")

    prompt_version_value = ""
    if "prompt_version" in prepared.columns:
        prompt_version_value = summarize_text_values(prepared["prompt_version"])

    prompt_parameters_value = ""
    prompt_parameters_metadata = None
    if "prompt_parameters_json" in prepared.columns:
        prompt_parameters_value, prompt_parameters_metadata = summarize_prompt_parameters(
            prepared["prompt_parameters_json"]
        )

    dataset_path = runtime_dir / "openlayer_input.csv"
    prepared.to_csv(dataset_path, index=False)

    output_root = runtime_dir / "openlayer_metric_outputs"
    dataset_output_dir = output_root / str(dataset_name)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    dataset_output_csv = dataset_output_dir / "dataset.csv"
    prepared.to_csv(dataset_output_csv, index=False)

    runtime_config = {
        "outputColumnName": "output",
        "inputVariableNames": input_variable_names,
        "contextColumnName": "context",
        "groundTruthColumnName": "ground_truth",
    }

    runtime_config_path = dataset_output_dir / "config.json"
    runtime_config_path.write_text(json.dumps(runtime_config, indent=2) + "\n", encoding="utf-8")

    dataset_cfg = {
        "name": str(dataset_name),
        "path": dataset_path.name,
        "label": "validation",
        "groundTruthColumnName": "ground_truth",
        "contextColumnName": "context",
        "inputVariableNames": input_variable_names,
        "outputColumnName": "output",
    }

    openlayer_cfg = {
        "taskType": "llm-base",
        "model": {
            "modelType": "shell",
            "outputDirectory": "openlayer_metric_outputs",
        },
        "datasets": [dataset_cfg],
    }

    openlayer_json_path = runtime_dir / "openlayer.json"
    openlayer_json_path.write_text(json.dumps(openlayer_cfg, indent=2) + "\n", encoding="utf-8")

    return {
        "openlayer_json_path": str(openlayer_json_path),
        "dataset_path": str(dataset_path),
        "dataset_output_csv": str(dataset_output_csv),
        "runtime_config_path": str(runtime_config_path),
        "prepared_rows": int(len(prepared)),
        "prepared_columns": list(prepared.columns),
        "input_variable_names": input_variable_names,
        "prompt_version": prompt_version_value,
        "prompt_parameters_json": prompt_parameters_value,
        "prompt_parameters_metadata": prompt_parameters_metadata,
    }


def openlayer_cli_download_url():
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system != "linux":
        raise Exception("This recipe expects Linux runtime in Dataiku. Detected: {}".format(system))

    if machine in ("x86_64", "amd64"):
        arch = "Linux_x86_64"
    elif machine in ("arm64", "aarch64"):
        arch = "Linux_arm64"
    else:
        raise Exception("Unsupported CPU architecture for OpenLayer CLI: {}".format(machine))

    return "https://downloads.openlayer.com/cli/download/latest/openlayer-cli_{}.tar.gz".format(arch)


def ensure_openlayer_cli():
    existing = shutil.which("openlayer")
    if existing:
        return existing

    home = Path(os.environ.get("HOME", "/tmp"))
    candidates = [
        home / ".openlayer" / "bin" / "openlayer",
        home / ".local" / "bin" / "openlayer",
        Path("/tmp/.local/bin/openlayer"),
        Path("/tmp/openlayer-cli/bin/openlayer"),
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    install_root = Path(os.environ.get("OPENLAYER_CLI_HOME", "/tmp/openlayer-cli"))
    bin_dir = install_root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    target_exe = bin_dir / "openlayer"

    if target_exe.is_file() and os.access(target_exe, os.X_OK):
        return str(target_exe)

    archive_url = openlayer_cli_download_url()
    archive_path = None

    try:
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".tar.gz") as tmpf:
            archive_path = tmpf.name
            with urlopen(archive_url, timeout=120) as response:
                tmpf.write(response.read())

        with tarfile.open(archive_path, "r:gz") as tar:
            member = None
            for m in tar.getmembers():
                if not m.isfile():
                    continue
                base = Path(m.name).name
                if base in ("openlayer-cli", "openlayer"):
                    member = m
                    break

            if member is None:
                raise Exception("Downloaded OpenLayer archive did not contain CLI binary")

            extracted = tar.extractfile(member)
            if extracted is None:
                raise Exception("Could not extract OpenLayer CLI binary from archive")

            with open(target_exe, "wb") as out:
                shutil.copyfileobj(extracted, out)

        os.chmod(target_exe, 0o755)

    finally:
        if archive_path:
            try:
                os.remove(archive_path)
            except OSError:
                pass

    if target_exe.is_file() and os.access(target_exe, os.X_OK):
        return str(target_exe)

    raise Exception("OpenLayer CLI install completed but executable was not found at {}".format(str(target_exe)))


def normalize_openlayer_base_url(api_url):
    parsed = urlparse(api_url.strip())
    netloc = parsed.netloc
    if netloc == "app.openlayer.com":
        netloc = "api.openlayer.com"

    path = parsed.path.rstrip("/")
    if not path:
        path = "/v1"

    return urlunparse(parsed._replace(netloc=netloc, path=path))


def ensure_openlayer_sdk():
    try:
        import openlayer as openlayer_module
        return openlayer_module
    except Exception:
        pass

    target = Path("/tmp/openlayer_sdk")
    target.mkdir(parents=True, exist_ok=True)

    if str(target) not in sys.path:
        sys.path.insert(0, str(target))

    try:
        import openlayer as openlayer_module
        return openlayer_module
    except Exception:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--upgrade",
                "--target",
                str(target),
                "openlayer>=0.20,<1.0",
            ]
        )
        importlib.invalidate_caches()
        import openlayer as openlayer_module
        return openlayer_module


def build_openlayer_client(api_key, api_url):
    openlayer_module = ensure_openlayer_sdk()
    kwargs = {"api_key": api_key}
    if api_url:
        kwargs["base_url"] = normalize_openlayer_base_url(api_url)
    return openlayer_module.Openlayer(**kwargs)


def first_present_attr(item, attr_names):
    for attr_name in attr_names:
        value = getattr(item, attr_name, None)
        if value is not None:
            return value
    return None


def serialize_test_result(item):
    goal = getattr(item, "goal", None)
    thresholds = []

    for threshold in getattr(goal, "thresholds", []) or []:
        thresholds.append(
            {
                "insightName": getattr(threshold, "insight_name", None),
                "measurement": getattr(threshold, "measurement", None),
                "condition": getattr(threshold, "condition", None),
                "value": getattr(threshold, "value", None),
            }
        )

    observed_value = first_present_attr(
        item,
        ["observed_value", "observedValue", "metric_value", "insight_value", "value", "score"],
    )

    return {
        "resultId": first_present_attr(item, ["id", "result_id"]),
        "observedValue": observed_value,
        "goal": {
            "name": getattr(goal, "name", None),
            "thresholds": thresholds,
        },
    }


def fetch_latest_project_version_id(client, project_id):
    versions = client.projects.commits.list(project_id=project_id, per_page=1)
    items = getattr(versions, "items", None) or []
    if not items:
        return ""
    return str(getattr(items[0], "id", "")).strip()


def fetch_test_results(client, project_version_id):
    last_error = None

    for _ in range(6):
        try:
            all_results = []
            page = 1
            per_page = 100

            while True:
                response = client.commits.test_results.list(
                    project_version_id=project_version_id,
                    page=page,
                    per_page=per_page,
                )
                items = getattr(response, "items", None) or []
                all_results.extend(items)

                if len(items) < per_page:
                    break
                page += 1

            if all_results:
                return [serialize_test_result(item) for item in all_results]

        except Exception as error:  # noqa: BLE001
            last_error = error

        time.sleep(5)

    if last_error:
        raise RuntimeError("Failed to fetch OpenLayer test results: {}".format(last_error))
    return []


def to_number(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def collect_numeric_paths(observed, path=None, out=None):
    if path is None:
        path = []
    if out is None:
        out = []

    numeric = to_number(observed)
    if numeric is not None:
        out.append((list(path), numeric))
        return out

    if isinstance(observed, dict):
        for key, value in observed.items():
            key_norm = sanitize_key(key)
            if key_norm in IGNORED_OBSERVED_KEYS:
                continue
            collect_numeric_paths(value, path + [key_norm], out)
        return out

    if isinstance(observed, list):
        for index, value in enumerate(observed):
            collect_numeric_paths(value, path + [str(index)], out)
        return out

    return out


def benchmark_tokens(benchmark_name, thresholds):
    tokens = set()

    bench_key = sanitize_key(benchmark_name)
    if bench_key:
        tokens.add(bench_key)
        for part in bench_key.split("_"):
            if len(part) >= 3:
                tokens.add(part)

    for threshold in thresholds:
        if not isinstance(threshold, dict):
            continue
        insight_name = sanitize_key(threshold.get("insightName"))
        if not insight_name:
            continue
        tokens.add(insight_name)
        for part in insight_name.split("_"):
            if len(part) >= 3:
                tokens.add(part)

    return tokens


def choose_benchmark_numeric_value(observed, benchmark_name, thresholds):
    numeric_paths = collect_numeric_paths(observed)
    if not numeric_paths:
        return None

    score_like_keys = {
        "score",
        "value",
        "overall",
        "overall_score",
        "mean",
        "avg",
        "metric_value",
        "observed_value",
        "observedvalue",
        "result",
    }
    noisy_keys = {"threshold", "thresholds", "condition", "target", "min", "max"}
    tokens = benchmark_tokens(benchmark_name, thresholds)

    ranked = []
    for path_parts, value in numeric_paths:
        path_joined = "_".join(path_parts)
        path_set = set(path_parts)

        score = 0
        for token in tokens:
            if token in path_set:
                score += 8
            elif token in path_joined:
                score += 4

        if path_parts and path_parts[-1] in score_like_keys:
            score += 3
        if any(part in noisy_keys for part in path_parts):
            score -= 5

        ranked.append((score, len(path_parts), value))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    best_score, _, best_value = ranked[0]

    if best_score > 0:
        return best_value

    for _, _, value in ranked:
        return value

    return None


def flatten_benchmark_columns(serialized_results):
    bucketed = {}

    for item in serialized_results:
        goal = item.get("goal") if isinstance(item.get("goal"), dict) else {}
        benchmark_name = str(goal.get("name") or "unnamed_benchmark")
        benchmark_key = sanitize_key(benchmark_name)
        benchmark_col = "benchmark_" + benchmark_key
        thresholds = goal.get("thresholds") if isinstance(goal.get("thresholds"), list) else []

        observed_value = item.get("observedValue")
        numeric_value = choose_benchmark_numeric_value(observed_value, benchmark_name, thresholds)
        if numeric_value is not None:
            if benchmark_col not in bucketed:
                bucketed[benchmark_col] = []
            bucketed[benchmark_col].append(float(numeric_value))

    flat = {}
    for benchmark_col, values in bucketed.items():
        if values:
            flat[benchmark_col] = sum(values) / len(values)

    return flat


# ------------------ main recipe flow ------------------ #

input_ds = dataiku.Dataset(INPUT_DATASET_NAME)
input_df = input_ds.get_dataframe()

variables = dataiku.get_custom_variables()
metadata_env = parse_env_text(variables.get("OPENLAYER_METADATA_ENV", ""))

api_key = str(variables.get("OPENLAYER_API_KEY", "")).strip()
project_id = str(variables.get("OPENLAYER_PROJECT_ID", "")).strip()
api_url = str(variables.get("OPENLAYER_API_URL", "")).strip()

if not api_key or not project_id:
    missing = []
    if not api_key:
        missing.append("OPENLAYER_API_KEY")
    if not project_id:
        missing.append("OPENLAYER_PROJECT_ID")
    raise Exception("Missing required Dataiku project variables: " + ", ".join(missing))

dataset_name = (
    metadata_env.get("DATAIKU_DATASET_NAME")
    or metadata_env.get("DATASET_NAME")
    or INPUT_DATASET_NAME
)

model_name = (
    metadata_env.get("OPENLAYER_LLM_NAME")
    or metadata_env.get("MODEL_NAME")
    or infer_model_name(input_df)
)

run_dt = datetime.now(timezone.utc)
run_date_utc = run_dt.date().isoformat()

dataset_version = (
    metadata_env.get("OPENLAYER_DATASET_VERSION")
    or metadata_env.get("DATASET_VERSION")
    or "{}@{}".format(dataset_name, run_date_utc)
)

push_message = "Dataiku OpenLayer run {} {} {}".format(
    dataset_name,
    model_name,
    run_dt.isoformat(),
)

run_env = os.environ.copy()
run_env["OPENLAYER_API_KEY"] = api_key
run_env["OPENLAYER_PROJECT_ID"] = project_id
if api_url:
    run_env["OPENLAYER_API_URL"] = api_url

run_env["OPENLAYER_LLM_NAME"] = model_name
run_env["OPENLAYER_DATASET_VERSION"] = dataset_version

run_suffix = uuid.uuid4().hex[:12]
runtime_dir = Path("/tmp/olr_" + run_suffix)
temp_dir = Path("/tmp/olt_" + run_suffix)
runtime_dir.mkdir(parents=True, exist_ok=True)
temp_dir.mkdir(parents=True, exist_ok=True)

run_env["TMPDIR"] = str(temp_dir)
run_env["TMP"] = str(temp_dir)
run_env["TEMP"] = str(temp_dir)
run_env.setdefault("HOME", "/tmp")

(runtime_dir / ".openlayer").mkdir(parents=True, exist_ok=True)
(runtime_dir / ".openlayer" / "config.json").write_text(
    json.dumps({"projectId": project_id}, indent=2) + "\n",
    encoding="utf-8",
)

prepared_runtime = prepare_openlayer_runtime_files(
    input_df=input_df,
    runtime_dir=runtime_dir,
    dataset_name=dataset_name,
)

prompt_version = (
    metadata_env.get("OPENLAYER_PROMPT_VERSION")
    or metadata_env.get("PROMPT_VERSION")
    or prepared_runtime.get("prompt_version", DEFAULT_PROMPT_VERSION)
)
prompt_parameters_json = prepared_runtime.get(
    "prompt_parameters_json",
    DEFAULT_PROMPT_PARAMETERS_JSON,
)
prompt_parameters_metadata = prepared_runtime.get("prompt_parameters_metadata")

if prompt_version and prompt_version != "mixed":
    run_env["OPENLAYER_PROMPT_VERSION"] = prompt_version

run_metadata_json = {}
if prompt_parameters_metadata:
    run_metadata_json["parameters"] = prompt_parameters_metadata

if run_metadata_json:
    run_env["OPENLAYER_RUN_METADATA_JSON"] = json.dumps(run_metadata_json, sort_keys=True)

openlayer_exe = ensure_openlayer_cli()
run_env["PATH"] = str(Path(openlayer_exe).parent) + ":" + run_env.get("PATH", "")

push_cmd = [
    openlayer_exe,
    "push",
    "--message",
    push_message,
    "--wait=true",
    "--api-key",
    api_key,
]
push_result = subprocess.run(
    push_cmd,
    text=True,
    capture_output=True,
    env=run_env,
    cwd=str(runtime_dir),
)

# Always inspect, even if push returned non-zero.
inspect_result = subprocess.run(
    [openlayer_exe, "inspect"],
    text=True,
    capture_output=True,
    env=run_env,
    cwd=str(runtime_dir),
)

inspect_stdout = inspect_result.stdout or ""
inspect_stderr = inspect_result.stderr or ""

inspect_status = extract_openlayer_status(inspect_stdout)
if inspect_status == "unknown":
    inspect_status = extract_openlayer_status(push_result.stdout)

# Fail only if processing itself did not complete.
if inspect_status != "completed":
    raise RuntimeError(
        "OpenLayer processing did not complete successfully.\n"
        "INSPECT_STATUS: {}\n"
        "PUSH_STDOUT:\n{}\n"
        "PUSH_STDERR:\n{}\n"
        "INSPECT_STDOUT:\n{}\n"
        "INSPECT_STDERR:\n{}\n".format(
            inspect_status,
            trim_text(push_result.stdout),
            trim_text(push_result.stderr),
            trim_text(inspect_stdout),
            trim_text(inspect_stderr),
        )
    )

project_version_id = (
    extract_project_version_id(inspect_stdout)
    or extract_project_version_id(push_result.stdout)
)

client = build_openlayer_client(api_key=api_key, api_url=api_url)
if not project_version_id:
    project_version_id = fetch_latest_project_version_id(client, project_id)

if not project_version_id:
    raise RuntimeError("Could not determine project version id for benchmark retrieval.")

serialized_results = fetch_test_results(client, project_version_id)
benchmark_columns = flatten_benchmark_columns(serialized_results)

if not benchmark_columns:
    raise RuntimeError("No numeric benchmark values found in OpenLayer observed results for this run.")

# Output only requested metadata + numeric benchmark values.
row = {
    "timestamp_utc": run_dt.isoformat(),
    "database_version": dataset_version,
    "llm_version": model_name,
    "prompt_version": prompt_version or DEFAULT_PROMPT_VERSION,
    "prompt_parameters_json": prompt_parameters_json or DEFAULT_PROMPT_PARAMETERS_JSON,
}
row.update(benchmark_columns)

output_df = pd.DataFrame([row])
fixed_cols = [
    "timestamp_utc",
    "database_version",
    "llm_version",
    "prompt_version",
    "prompt_parameters_json",
]
ordered_cols = fixed_cols + [c for c in output_df.columns if c not in fixed_cols]
output_df = output_df[ordered_cols]

# One row per run. In Dataiku recipe settings set output mode to Append to keep history.
output_ds = dataiku.Dataset(OUTPUT_DATASET_NAME)
output_ds.write_with_schema(output_df)
