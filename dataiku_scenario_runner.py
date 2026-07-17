"""Run OpenLayer pipeline from Dataiku and save benchmark logs for analysis.

This script is designed for Dataiku scenario steps (Python or command) and can be
kept in the repository as a single runner.

Workflow:
1. Prepare dataset for OpenLayer
2. Render OpenLayer config profile
3. Push artifacts with OpenLayer CLI
4. Wait for processing and write JSON log(s)
5. Export OpenLayer logs into flat CSV tables
6. Build latest and cumulative history CSV outputs
7. Optionally upload outputs to a Dataiku managed folder
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

RUN_DEDUPLICATION_KEYS = ["project_version_id", "log_file"]
BENCHMARK_DEDUPLICATION_KEYS = [
    "project_version_id",
    "item_index",
    "threshold_index",
    "result_id",
    "benchmark_name",
    "measurement",
]


class PipelineRunError(RuntimeError):
    """Raised when the OpenLayer pipeline runner cannot continue."""


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenLayer CLI pipeline from Dataiku and persist benchmark outputs.",
    )

    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root path (default: current working directory).",
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--openlayer-exe", default="openlayer")

    parser.add_argument("--env-file", default="")
    parser.add_argument("--metadata-env-file", default="")

    parser.add_argument("--input-csv", type=Path, default=Path("data/richcalls_final.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/processed/output_long_format.csv"))
    parser.add_argument("--per-model-output-dir", type=Path, default=Path("data/processed/by_model"))
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--profile", default="summaries")
    parser.add_argument("--push-message", default="")

    parser.add_argument("--api-key", default="")
    parser.add_argument("--project-id", default="")
    parser.add_argument("--api-url", default="")
    parser.add_argument("--run-metadata-path", default="")

    parser.add_argument("--execution-source", default="dataiku")
    parser.add_argument("--execution-trigger", default="scenario")
    parser.add_argument("--push-custom-metrics", action="store_true")

    parser.add_argument("--log-dir", type=Path, default=Path("logs/openlayer"))
    parser.add_argument(
        "--runs-output",
        type=Path,
        default=Path("data/processed/openlayer/openlayer_runs.csv"),
    )
    parser.add_argument(
        "--benchmarks-output",
        type=Path,
        default=Path("data/processed/openlayer/openlayer_benchmark_thresholds.csv"),
    )
    parser.add_argument(
        "--easy-summary-output",
        type=Path,
        default=Path("data/processed/openlayer/openlayer_easy_summary_latest.json"),
    )

    parser.add_argument(
        "--latest-runs-output",
        type=Path,
        default=Path("data/processed/openlayer/latest/openlayer_runs_latest.csv"),
    )
    parser.add_argument(
        "--latest-benchmarks-output",
        type=Path,
        default=Path("data/processed/openlayer/latest/openlayer_benchmark_thresholds_latest.csv"),
    )
    parser.add_argument(
        "--history-runs-output",
        type=Path,
        default=Path("data/processed/openlayer/history/openlayer_runs_history.csv"),
    )
    parser.add_argument(
        "--history-benchmarks-output",
        type=Path,
        default=Path("data/processed/openlayer/history/openlayer_benchmark_thresholds_history.csv"),
    )
    parser.add_argument(
        "--stack-mode",
        choices=["latest", "full"],
        default="latest",
        help="latest: append only newest project_version_id, full: append entire exports",
    )

    parser.add_argument("--managed-folder-id", default="")
    parser.add_argument("--managed-folder-prefix", default="openlayer")

    return parser.parse_args(list(argv) if argv is not None else None)


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return repo_root / candidate


def load_env_file(path: Path, target_env: dict[str, str]) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or not value:
            continue

        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        if value:
            target_env[key] = value


def run_command_no_capture(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    context: str,
) -> None:
    print(f"[run] {context}: {' '.join(command)}")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def run_command_capture(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    context: str,
) -> subprocess.CompletedProcess[str]:
    print(f"[run] {context}: {' '.join(command)}")
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )


def resolve_openlayer_command(openlayer_exe: str, python_exe: str) -> list[str]:
    """Resolve how to invoke OpenLayer CLI in varied runtimes.

    Priority:
    1. Explicit --openlayer-exe path/command when provided
    2. openlayer binary discoverable in PATH
    3. python -m openlayer fallback
    """
    candidate = openlayer_exe.strip()
    if candidate:
        # If a full path is provided, use it directly.
        if Path(candidate).is_file():
            return [candidate]

        # If a command name is provided and available in PATH, use it.
        resolved = shutil.which(candidate)
        if resolved:
            return [resolved]

    resolved_default = shutil.which("openlayer")
    if resolved_default:
        return [resolved_default]

    return [python_exe, "-m", "openlayer"]


def safe_git_value(repo_root: Path, args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def ensure_openlayer_project_config(repo_root: Path, project_id: str) -> None:
    config_dir = repo_root / ".openlayer"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps({"projectId": project_id}, indent=2) + "\n", encoding="utf-8")


def load_project_id_from_openlayer_config(repo_root: Path) -> str:
    config_path = repo_root / ".openlayer" / "config.json"
    if not config_path.is_file():
        return ""

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return ""

    if not isinstance(payload, dict):
        return ""

    project_id = payload.get("projectId")
    return str(project_id).strip() if project_id is not None else ""


def ensure_not_placeholder(name: str, value: str) -> str:
    text = value.strip()
    placeholders = {
        "YOUR_REAL_API_KEY",
        "<your_api_key>",
        "PASTE_OPENLAYER_API_KEY_HERE",
        "<your_project_id>",
    }
    if not text:
        raise PipelineRunError(f"{name} is required")
    if text in placeholders:
        raise PipelineRunError(f"{name} still contains a placeholder value")
    return text


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        line_count = sum(1 for _ in handle)
    return max(line_count - 1, 0)


def compact_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def infer_model_name_from_input(input_csv: Path) -> str:
    try:
        sample = pd.read_csv(input_csv, nrows=500)
    except Exception as error:  # noqa: BLE001
        print(f"Warning: unable to infer model name from input CSV: {error}")
        return "unknown-model"

    candidate_columns = [
        "model_name",
        "model",
        "llm_name",
        "model_used",
        "generator_model",
        "model_id",
    ]

    for column in candidate_columns:
        if column not in sample.columns:
            continue

        values = [
            str(value).strip()
            for value in sample[column].dropna().astype(str).tolist()
            if str(value).strip()
        ]
        unique_values = sorted(set(values))
        if not unique_values:
            continue
        if len(unique_values) == 1:
            return unique_values[0]
        preview = ",".join(unique_values[:3])
        if len(unique_values) > 3:
            preview += ",..."
        return f"multi:{preview}"

    model_columns = sorted(
        [column for column in sample.columns if column.startswith("summary_model_")]
    )
    if model_columns:
        labels = [column.removeprefix("summary_") for column in model_columns]
        if len(labels) == 1:
            return labels[0]
        return "multi:" + ",".join(labels)

    return "unknown-model"


def build_easy_summary(
    runs_output: Path,
    benchmarks_output: Path,
    summary_output: Path,
    dataset_name: str,
    model_name: str,
) -> dict[str, Any]:
    runs_frame = read_csv_required(runs_output, "runs")
    runs_frame["generated_at"] = pd.to_datetime(runs_frame.get("generated_at"), utc=True, errors="coerce")
    runs_frame = runs_frame.sort_values("generated_at", ascending=False, na_position="last")
    latest_row = runs_frame.iloc[0].to_dict()

    latest_project_version_id = compact_text(latest_row.get("project_version_id"), "unknown")
    benchmarks_frame = read_csv_optional(benchmarks_output)
    latest_benchmarks = pd.DataFrame()

    if not benchmarks_frame.empty and "project_version_id" in benchmarks_frame.columns:
        latest_benchmarks = benchmarks_frame[
            benchmarks_frame["project_version_id"].astype(str).str.strip().eq(latest_project_version_id)
        ].copy()

    benchmark_status_counts: dict[str, int] = {}
    failing_benchmarks: list[dict[str, Any]] = []
    if not latest_benchmarks.empty:
        if "status" in latest_benchmarks.columns:
            status_series = latest_benchmarks["status"].fillna("unknown").astype(str).str.lower()
            counts = status_series.value_counts()
            benchmark_status_counts = {str(key): int(value) for key, value in counts.items()}
            non_passing = latest_benchmarks[~status_series.eq("passing")]
        else:
            non_passing = latest_benchmarks.copy()

        selected_columns = [
            "benchmark_name",
            "status",
            "status_message",
            "measurement",
            "threshold_value",
            "observed_value",
        ]
        available_columns = [column for column in selected_columns if column in non_passing.columns]
        if available_columns:
            failing_benchmarks = non_passing[available_columns].head(20).to_dict(orient="records")

    summary_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "model_name": model_name,
        "latest_run": {
            "project_version_id": latest_project_version_id,
            "generated_at": str(latest_row.get("generated_at", "")),
            "processing_status": compact_text(latest_row.get("processing_status"), "unknown"),
            "item_count": int(latest_row.get("item_count", 0) or 0),
            "passing_count": int(latest_row.get("item_status_passing", 0) or 0),
            "failing_count": int(latest_row.get("item_status_failing", 0) or 0),
            "error_count": int(latest_row.get("item_status_error", 0) or 0),
            "profile_name": compact_text(latest_row.get("profile_name"), "unknown"),
        },
        "benchmark_status_counts": benchmark_status_counts,
        "failing_benchmarks": failing_benchmarks,
    }

    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    print("Easy summary:")
    print(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "project_version_id": latest_project_version_id,
                "processing_status": summary_payload["latest_run"]["processing_status"],
                "benchmark_status_counts": benchmark_status_counts,
                "summary_file": str(summary_output),
            },
            indent=2,
        )
    )

    return summary_payload


def tests_require_custom_metrics(tests_path: Path) -> bool:
    if not tests_path.is_file():
        return False

    try:
        payload = json.loads(tests_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if not isinstance(payload, list):
        return False

    for test in payload:
        if not isinstance(test, dict):
            continue

        if str(test.get("subtype", "")).strip() == "customMetricThreshold":
            return True

        thresholds = test.get("thresholds") if isinstance(test.get("thresholds"), list) else []
        for threshold in thresholds:
            if not isinstance(threshold, dict):
                continue
            insight_name = str(threshold.get("insightName", "")).strip()
            if insight_name in {"metrics", "customMetric"}:
                return True

    return False


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_csv_required(path: Path, table_name: str) -> pd.DataFrame:
    if not path.is_file():
        raise PipelineRunError(f"Missing {table_name} input CSV: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise PipelineRunError(f"Input CSV is empty for {table_name}: {path}")

    return frame


def parse_generated_at(frame: pd.DataFrame) -> pd.Series:
    if "generated_at" not in frame.columns:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(frame["generated_at"], utc=True, errors="coerce")


def pick_latest_project_version_id(runs_frame: pd.DataFrame) -> str:
    if "project_version_id" not in runs_frame.columns:
        raise PipelineRunError("runs export is missing required column: project_version_id")

    with_timestamps = runs_frame.copy()
    with_timestamps["_generated_at_ts"] = parse_generated_at(with_timestamps)
    with_timestamps = with_timestamps.sort_values("_generated_at_ts", ascending=False, na_position="last")

    for raw_value in with_timestamps["project_version_id"].tolist():
        value = str(raw_value).strip()
        if value and value.lower() != "nan":
            return value

    raise PipelineRunError("No usable project_version_id found in runs export")


def filter_latest_slice(
    runs_frame: pd.DataFrame,
    benchmarks_frame: pd.DataFrame,
    latest_project_version_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_runs = runs_frame[
        runs_frame["project_version_id"].astype(str).str.strip().eq(latest_project_version_id)
    ].copy()
    if latest_runs.empty:
        raise PipelineRunError(
            "Latest project version id could not be matched back to runs rows: "
            f"{latest_project_version_id}"
        )

    if "project_version_id" in benchmarks_frame.columns:
        latest_benchmarks = benchmarks_frame[
            benchmarks_frame["project_version_id"].astype(str).str.strip().eq(latest_project_version_id)
        ].copy()
    else:
        latest_benchmarks = benchmarks_frame.copy()

    return latest_runs, latest_benchmarks


def add_stack_metadata(
    frame: pd.DataFrame,
    stack_batch_id: str,
    stack_timestamp: str,
    stack_mode: str,
) -> pd.DataFrame:
    output = frame.copy()
    output["stack_batch_id"] = stack_batch_id
    output["stacked_at_utc"] = stack_timestamp
    output["stack_mode"] = stack_mode
    return output


def merge_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return incoming.copy()
    if incoming.empty:
        return existing.copy()

    all_columns = list(dict.fromkeys([*existing.columns.tolist(), *incoming.columns.tolist()]))
    existing_aligned = existing.reindex(columns=all_columns)
    incoming_aligned = incoming.reindex(columns=all_columns)
    return pd.concat([existing_aligned, incoming_aligned], ignore_index=True, sort=False)


def deduplicate_frame(frame: pd.DataFrame, preferred_keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame

    usable_keys = [column for column in preferred_keys if column in frame.columns]
    if not usable_keys:
        return frame.drop_duplicates(keep="last")

    return frame.drop_duplicates(subset=usable_keys, keep="last")


def sort_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame

    sorted_frame = frame.copy()

    if "generated_at" in sorted_frame.columns:
        sorted_frame["_generated_at_ts"] = pd.to_datetime(
            sorted_frame["generated_at"],
            utc=True,
            errors="coerce",
        )
    else:
        sorted_frame["_generated_at_ts"] = pd.NaT

    sorted_frame["_stacked_at_ts"] = pd.to_datetime(
        sorted_frame.get("stacked_at_utc"),
        utc=True,
        errors="coerce",
    )

    sorted_frame = sorted_frame.sort_values(
        ["_generated_at_ts", "_stacked_at_ts"],
        ascending=[False, False],
        na_position="last",
    )

    return sorted_frame.drop(columns=["_generated_at_ts", "_stacked_at_ts"], errors="ignore")


def write_csv(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def stack_exports(
    runs_input: Path,
    benchmarks_input: Path,
    latest_runs_output: Path,
    latest_benchmarks_output: Path,
    history_runs_output: Path,
    history_benchmarks_output: Path,
    mode: str,
) -> dict[str, Any]:
    runs_frame = read_csv_required(runs_input, "runs")
    benchmarks_frame = read_csv_optional(benchmarks_input)

    if mode == "latest":
        latest_project_version_id = pick_latest_project_version_id(runs_frame)
        incoming_runs, incoming_benchmarks = filter_latest_slice(
            runs_frame=runs_frame,
            benchmarks_frame=benchmarks_frame,
            latest_project_version_id=latest_project_version_id,
        )
    else:
        latest_project_version_id = ""
        incoming_runs = runs_frame.copy()
        incoming_benchmarks = benchmarks_frame.copy()

    stack_batch_id = str(uuid.uuid4())
    stack_timestamp = datetime.now(timezone.utc).isoformat()

    incoming_runs = add_stack_metadata(
        incoming_runs,
        stack_batch_id=stack_batch_id,
        stack_timestamp=stack_timestamp,
        stack_mode=mode,
    )
    incoming_benchmarks = add_stack_metadata(
        incoming_benchmarks,
        stack_batch_id=stack_batch_id,
        stack_timestamp=stack_timestamp,
        stack_mode=mode,
    )

    existing_runs = read_csv_optional(history_runs_output)
    existing_benchmarks = read_csv_optional(history_benchmarks_output)

    merged_runs = merge_frames(existing_runs, incoming_runs)
    merged_benchmarks = merge_frames(existing_benchmarks, incoming_benchmarks)

    deduplicated_runs = deduplicate_frame(merged_runs, RUN_DEDUPLICATION_KEYS)
    deduplicated_benchmarks = deduplicate_frame(merged_benchmarks, BENCHMARK_DEDUPLICATION_KEYS)

    history_runs = sort_history(deduplicated_runs)
    history_benchmarks = sort_history(deduplicated_benchmarks)
    latest_runs = sort_history(incoming_runs)
    latest_benchmarks = sort_history(incoming_benchmarks)

    write_csv(latest_runs, latest_runs_output)
    write_csv(latest_benchmarks, latest_benchmarks_output)
    write_csv(history_runs, history_runs_output)
    write_csv(history_benchmarks, history_benchmarks_output)

    return {
        "stack_batch_id": stack_batch_id,
        "stacked_at_utc": stack_timestamp,
        "mode": mode,
        "latest_project_version_id": latest_project_version_id or None,
        "incoming": {
            "runs": int(len(incoming_runs)),
            "benchmarks": int(len(incoming_benchmarks)),
        },
        "history": {
            "runs": int(len(history_runs)),
            "benchmarks": int(len(history_benchmarks)),
        },
    }


def upload_outputs_to_managed_folder(
    managed_folder_id: str,
    managed_folder_prefix: str,
    output_paths: list[Path],
) -> None:
    try:
        import dataiku  # type: ignore
    except Exception as error:  # noqa: BLE001
        raise PipelineRunError(
            "Managed-folder upload requested but dataiku package is unavailable"
        ) from error

    folder = dataiku.Folder(managed_folder_id)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for output_path in output_paths:
        if not output_path.is_file():
            raise PipelineRunError(f"Output not found for upload: {output_path}")

        latest_target = f"{managed_folder_prefix}/latest/{output_path.name}"
        run_target = f"{managed_folder_prefix}/history/{run_stamp}/{output_path.name}"

        with output_path.open("rb") as stream:
            folder.upload_stream(latest_target, stream)
        with output_path.open("rb") as stream:
            folder.upload_stream(run_target, stream)

        print(f"[upload] {output_path} -> {latest_target}")
        print(f"[upload] {output_path} -> {run_target}")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    repo_root = args.repo_root.resolve()
    if not repo_root.is_dir():
        print(f"Error: repo root not found: {repo_root}")
        return 2

    runtime_env = os.environ.copy()
    openlayer_command = resolve_openlayer_command(args.openlayer_exe, args.python_exe)

    if args.env_file:
        load_env_file(resolve_repo_path(repo_root, args.env_file), runtime_env)
    if args.metadata_env_file:
        load_env_file(resolve_repo_path(repo_root, args.metadata_env_file), runtime_env)

    api_key = args.api_key.strip() or runtime_env.get("OPENLAYER_API_KEY", "").strip()
    project_id = args.project_id.strip() or runtime_env.get("OPENLAYER_PROJECT_ID", "").strip()
    api_url = args.api_url.strip() or runtime_env.get("OPENLAYER_API_URL", "").strip()

    if not project_id:
        project_id = load_project_id_from_openlayer_config(repo_root)

    try:
        api_key = ensure_not_placeholder("OPENLAYER_API_KEY", api_key)
        project_id = ensure_not_placeholder("OPENLAYER_PROJECT_ID", project_id)
    except PipelineRunError as error:
        print(f"Error: {error}")
        return 2

    input_csv = resolve_repo_path(repo_root, args.input_csv)
    output_csv = resolve_repo_path(repo_root, args.output_csv)
    per_model_output_dir = resolve_repo_path(repo_root, args.per_model_output_dir)

    log_dir = resolve_repo_path(repo_root, args.log_dir)
    runs_output = resolve_repo_path(repo_root, args.runs_output)
    benchmarks_output = resolve_repo_path(repo_root, args.benchmarks_output)
    easy_summary_output = resolve_repo_path(repo_root, args.easy_summary_output)
    latest_runs_output = resolve_repo_path(repo_root, args.latest_runs_output)
    latest_benchmarks_output = resolve_repo_path(repo_root, args.latest_benchmarks_output)
    history_runs_output = resolve_repo_path(repo_root, args.history_runs_output)
    history_benchmarks_output = resolve_repo_path(repo_root, args.history_benchmarks_output)

    if not input_csv.is_file():
        print(f"Error: input CSV not found: {input_csv}")
        return 2

    dataset_name = args.dataset_name.strip() or runtime_env.get("DATAIKU_DATASET_NAME", "").strip() or input_csv.stem
    model_name = args.model_name.strip() or runtime_env.get("OPENLAYER_LLM_NAME", "").strip()
    if not model_name:
        model_name = infer_model_name_from_input(input_csv)

    run_timestamp_utc = datetime.now(timezone.utc)
    run_date_utc = run_timestamp_utc.date().isoformat()

    push_message = args.push_message.strip() or (
        f"Dataiku OpenLayer run {dataset_name} {model_name} {run_timestamp_utc.isoformat()}"
    )

    runtime_env["OPENLAYER_API_KEY"] = api_key
    runtime_env["OPENLAYER_PROJECT_ID"] = project_id

    if api_url:
        runtime_env["OPENLAYER_API_URL"] = api_url
    else:
        runtime_env.pop("OPENLAYER_API_URL", None)

    runtime_env["OPENLAYER_EXECUTION_SOURCE"] = args.execution_source
    runtime_env["OPENLAYER_EXECUTION_TRIGGER"] = args.execution_trigger
    runtime_env.setdefault("OPENLAYER_EXECUTION_ACTOR", os.environ.get("USER", "dataiku"))
    runtime_env.setdefault("OPENLAYER_EXECUTION_HOST", platform.node())
    runtime_env["OPENLAYER_PROFILE_NAME"] = args.profile
    runtime_env["OPENLAYER_LLM_NAME"] = model_name
    runtime_env.setdefault("OPENLAYER_LLM_PROVIDER", "dataiku")
    runtime_env["OPENLAYER_PUSH_MESSAGE"] = push_message
    runtime_env["OPENLAYER_DATASET_INPUT_PATH"] = str(input_csv)
    runtime_env["OPENLAYER_DATASET_OUTPUT_PATH"] = str(output_csv)
    runtime_env.setdefault("OPENLAYER_DATASET_VERSION", f"{dataset_name}@{run_date_utc}")
    runtime_env["OPENLAYER_PARAM_dataset_name"] = dataset_name
    runtime_env["OPENLAYER_PARAM_model_name"] = model_name
    runtime_env["OPENLAYER_PARAM_run_date_utc"] = run_date_utc
    runtime_env["OPENLAYER_PARAM_run_timestamp_utc"] = run_timestamp_utc.isoformat()

    if args.run_metadata_path:
        run_metadata_path = resolve_repo_path(repo_root, args.run_metadata_path)
        if not run_metadata_path.is_file():
            print(f"Error: run metadata file not found: {run_metadata_path}")
            return 2
        runtime_env["OPENLAYER_RUN_METADATA_PATH"] = str(run_metadata_path)

    git_branch = safe_git_value(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    git_commit = safe_git_value(repo_root, ["rev-parse", "HEAD"])
    remote_url = safe_git_value(repo_root, ["config", "--get", "remote.origin.url"])

    if git_branch:
        runtime_env["OPENLAYER_GIT_BRANCH"] = git_branch
    if git_commit:
        runtime_env["OPENLAYER_GIT_COMMIT"] = git_commit

    if "github.com" in remote_url:
        suffix = remote_url.split("github.com")[-1].lstrip(":/")
        if suffix.endswith(".git"):
            suffix = suffix[:-4]
        runtime_env["OPENLAYER_REPO_SLUG"] = suffix

    try:
        version_result = run_command_capture(
            [*openlayer_command, "--version"],
            cwd=repo_root,
            env=runtime_env,
            context="read OpenLayer CLI version",
        )
        version_lines = [line.strip() for line in version_result.stdout.splitlines() if line.strip()]
        if version_lines:
            runtime_env["OPENLAYER_CLI_VERSION"] = version_lines[-1]
    except Exception as error:  # noqa: BLE001
        print(f"Warning: unable to read OpenLayer CLI version: {error}")

    ensure_openlayer_project_config(repo_root, project_id)

    try:
        run_command_no_capture(
            [
                args.python_exe,
                "src/prepare_openlayer_dataset.py",
                "--input-csv",
                str(input_csv),
                "--output-csv",
                str(output_csv),
                "--per-model-output-dir",
                str(per_model_output_dir),
            ],
            cwd=repo_root,
            env=runtime_env,
            context="prepare OpenLayer dataset",
        )

        if output_csv.is_file():
            runtime_env["OPENLAYER_DATASET_SHA256"] = compute_sha256(output_csv)
            runtime_env["OPENLAYER_DATASET_ROWS"] = str(count_csv_rows(output_csv))

        run_command_no_capture(
            [args.python_exe, "scripts/render_openlayer_config.py", "--profile", args.profile],
            cwd=repo_root,
            env=runtime_env,
            context="render OpenLayer profile",
        )

        tests_path = repo_root / "tests.json"
        if tests_require_custom_metrics(tests_path):
            run_command_no_capture(
                [
                    args.python_exe,
                    "scripts/prepare_openlayer_metric_inputs.py",
                    "--openlayer-config",
                    "openlayer.json",
                ],
                cwd=repo_root,
                env=runtime_env,
                context="prepare custom metric runtime inputs",
            )

            if args.push_custom_metrics:
                metrics_dir = repo_root / "metrics"
                if metrics_dir.is_dir():
                    run_command_no_capture(
                        [
                            *openlayer_command,
                            "metrics",
                            "push",
                            "-d",
                            "metrics",
                            "--api-key",
                            api_key,
                        ],
                        cwd=repo_root,
                        env=runtime_env,
                        context="push custom metric definitions",
                    )
                else:
                    print("Warning: custom metric push requested but metrics directory was not found")
            else:
                print(
                    "Info: custom metrics detected; skipping metric-definition push by default. "
                    "Use --push-custom-metrics when metric code/config changed."
                )
        else:
            print("Info: tests.json does not require custom metric thresholds; skipping metric prep")

        run_command_no_capture(
            [
                *openlayer_command,
                "push",
                "--message",
                push_message,
                "--wait=false",
                "--api-key",
                api_key,
            ],
            cwd=repo_root,
            env=runtime_env,
            context="push artifacts to OpenLayer",
        )

        run_command_no_capture(
            [args.python_exe, "scripts/wait_for_openlayer_processing.py"],
            cwd=repo_root,
            env=runtime_env,
            context="wait for OpenLayer processing",
        )

        run_command_no_capture(
            [
                args.python_exe,
                "src/export_openlayer_logs_for_dataiku.py",
                "--log-dir",
                str(log_dir),
                "--runs-output",
                str(runs_output),
                "--benchmarks-output",
                str(benchmarks_output),
            ],
            cwd=repo_root,
            env=runtime_env,
            context="export OpenLayer logs to flat CSV tables",
        )

        stack_summary = stack_exports(
            runs_input=runs_output,
            benchmarks_input=benchmarks_output,
            latest_runs_output=latest_runs_output,
            latest_benchmarks_output=latest_benchmarks_output,
            history_runs_output=history_runs_output,
            history_benchmarks_output=history_benchmarks_output,
            mode=args.stack_mode,
        )

        easy_summary = build_easy_summary(
            runs_output=runs_output,
            benchmarks_output=benchmarks_output,
            summary_output=easy_summary_output,
            dataset_name=dataset_name,
            model_name=model_name,
        )

        output_files = [
            runs_output,
            benchmarks_output,
            easy_summary_output,
            latest_runs_output,
            latest_benchmarks_output,
            history_runs_output,
            history_benchmarks_output,
        ]

        if args.managed_folder_id.strip():
            upload_outputs_to_managed_folder(
                managed_folder_id=args.managed_folder_id.strip(),
                managed_folder_prefix=args.managed_folder_prefix.strip("/"),
                output_paths=output_files,
            )

        summary = {
            "status": "ok",
            "repo_root": str(repo_root),
            "dataset_name": dataset_name,
            "model_name": model_name,
            "run_date_utc": run_date_utc,
            "profile": args.profile,
            "stack_mode": args.stack_mode,
            "managed_folder_upload": bool(args.managed_folder_id.strip()),
            "stack": stack_summary,
            "easy_summary": easy_summary,
            "outputs": [str(path) for path in output_files],
        }
        print(json.dumps(summary, indent=2))
        return 0

    except subprocess.CalledProcessError as error:
        print(f"Error: command failed with exit code {error.returncode}")
        if error.stdout:
            print("[stdout]")
            print(error.stdout)
        if error.stderr:
            print("[stderr]")
            print(error.stderr)
        return error.returncode if error.returncode else 1
    except FileNotFoundError as error:
        print("Error: OpenLayer CLI executable was not found in this runtime.")
        print("Install OpenLayer in the selected Dataiku code environment or provide --openlayer-exe.")
        print(f"Details: {error}")
        return 2
    except PipelineRunError as error:
        print(f"Error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
