from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = ROOT / "data"
SNAPSHOT_ROOT = DATA_ROOT / "snapshots"
RUN_ROOT = DATA_ROOT / "runs"
BACKTEST_ROOT = DATA_ROOT / "backtests"

INTELLIGENCE_VERSION = "1.0.0"
ARCHIVE_SCHEMA_VERSION = "1.0"

REQUIRED_POOL_COLUMNS = {
    "player_id",
    "name",
    "position",
    "team",
    "opponent",
    "salary",
    "roster_positions",
    "status",
    "projection",
}

SITE_CAPS = {
    "DraftKings": 50000,
    "FanDuel": 60000,
}


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )


def _safe_slug(value: str) -> str:
    cleaned = "".join(
        char.lower()
        if char.isalnum()
        else "-"
        for char in str(value)
    )

    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")

    return cleaned.strip("-") or "snapshot"


def _json_default(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, set):
        return sorted(value)

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable."
    )


def _atomic_json_write(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )

    temp.replace(path)


def _read_json(
    path: Path,
    default=None,
):
    if not path.exists():
        return default

    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def git_commit() -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def inspect_player_pool(
    pool_file: Path,
    site_key: str,
    site_name: str,
) -> dict[str, Any]:
    pool_file = Path(pool_file)

    blocking_errors: list[str] = []
    warnings: list[str] = []

    report: dict[str, Any] = {
        "site_key": site_key,
        "site_name": site_name,
        "salary_cap": SITE_CAPS.get(
            site_name
        ),
        "pool_file": str(pool_file),
        "exists": pool_file.exists(),
        "row_count": 0,
        "sha256": None,
        "file_modified_utc": None,
        "status_counts": {},
        "position_counts": {},
        "blocking_errors": blocking_errors,
        "warnings": warnings,
        "state": "BLOCKED",
    }

    if not pool_file.exists():
        blocking_errors.append(
            "Player pool file does not exist."
        )
        return report

    stat = pool_file.stat()

    report["file_modified_utc"] = (
        datetime.fromtimestamp(
            stat.st_mtime,
            timezone.utc,
        )
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    report["sha256"] = sha256_file(
        pool_file
    )

    try:
        handle = pool_file.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        )
    except Exception as exc:
        blocking_errors.append(
            f"Could not open player pool: {exc}"
        )
        return report

    with handle:
        reader = csv.DictReader(handle)

        columns = set(
            reader.fieldnames or []
        )

        missing_columns = sorted(
            REQUIRED_POOL_COLUMNS - columns
        )

        if missing_columns:
            blocking_errors.append(
                "Missing required columns: "
                + ", ".join(
                    missing_columns
                )
            )

            return report

        ids: set[str] = set()
        duplicate_ids: set[str] = set()

        status_counts = Counter()
        position_counts = Counter()

        invalid_salary_rows = 0
        invalid_projection_rows = 0
        missing_identity_rows = 0
        blank_opponents = 0

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            report["row_count"] += 1

            player_id = str(
                row.get("player_id", "")
            ).strip()

            name = str(
                row.get("name", "")
            ).strip()

            position = str(
                row.get("position", "")
            ).strip().upper()

            team = str(
                row.get("team", "")
            ).strip().upper()

            opponent = str(
                row.get("opponent", "")
            ).strip().upper()

            status = str(
                row.get("status", "")
            ).strip().upper()

            if not (
                player_id
                and name
                and position
                and team
            ):
                missing_identity_rows += 1

            if player_id:
                if player_id in ids:
                    duplicate_ids.add(
                        player_id
                    )

                ids.add(player_id)

            if not opponent:
                blank_opponents += 1

            try:
                salary = float(
                    row.get(
                        "salary",
                        "",
                    )
                )

                if salary <= 0:
                    raise ValueError
            except Exception:
                invalid_salary_rows += 1

            try:
                float(
                    row.get(
                        "projection",
                        "",
                    )
                )
            except Exception:
                invalid_projection_rows += 1

            status_counts[
                status or "ACTIVE/BLANK"
            ] += 1

            position_counts[
                position or "UNKNOWN"
            ] += 1

    if report["row_count"] == 0:
        blocking_errors.append(
            "Player pool contains zero rows."
        )

    if duplicate_ids:
        preview = ", ".join(
            sorted(duplicate_ids)[:10]
        )

        blocking_errors.append(
            "Duplicate player IDs detected: "
            + preview
        )

    if missing_identity_rows:
        blocking_errors.append(
            f"{missing_identity_rows} rows "
            "are missing player identity fields."
        )

    if invalid_salary_rows:
        blocking_errors.append(
            f"{invalid_salary_rows} rows "
            "contain invalid salaries."
        )

    if invalid_projection_rows:
        blocking_errors.append(
            f"{invalid_projection_rows} rows "
            "contain invalid projections."
        )

    nonblank_status_count = sum(
        count
        for status, count
        in status_counts.items()
        if status != "ACTIVE/BLANK"
    )

    if nonblank_status_count:
        warnings.append(
            f"{nonblank_status_count} players "
            "have a nonblank status and require "
            "game-day review."
        )

    if blank_opponents:
        warnings.append(
            f"{blank_opponents} rows have "
            "a blank opponent field."
        )

    report["status_counts"] = dict(
        sorted(status_counts.items())
    )

    report["position_counts"] = dict(
        sorted(position_counts.items())
    )

    report["blank_opponent_count"] = (
        blank_opponents
    )

    if blocking_errors:
        report["state"] = "BLOCKED"
    elif warnings:
        report["state"] = "REVIEW"
    else:
        report["state"] = "READY"

    return report


def readiness_report(
    site_config: dict,
) -> dict[str, Any]:
    sites = {}

    for site_key, config in site_config.items():
        sites[site_key] = inspect_player_pool(
            Path(config["pool_file"]),
            site_key=site_key,
            site_name=config["site_name"],
        )

    states = {
        site["state"]
        for site in sites.values()
    }

    if "BLOCKED" in states:
        overall = "BLOCKED"
    elif "REVIEW" in states:
        overall = "REVIEW"
    else:
        overall = "READY"

    return {
        "intelligence_version": (
            INTELLIGENCE_VERSION
        ),
        "generated_at": utc_now_iso(),
        "overall_state": overall,
        "freshness_policy": (
            "Structural integrity is verified. "
            "External injury/projection freshness "
            "must still be validated before lock."
        ),
        "sites": sites,
    }


def capture_slate_snapshot(
    site_config: dict,
    slate: str,
    label: str = "manual",
) -> dict[str, Any]:
    SNAPSHOT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapshot_id = (
        f"{_timestamp_slug()}_"
        f"{_safe_slug(slate)}_"
        f"{_safe_slug(label)}_"
        f"{uuid4().hex[:8]}"
    )

    target = (
        SNAPSHOT_ROOT
        / snapshot_id
    )

    target.mkdir(
        parents=True,
        exist_ok=False,
    )

    sites = {}

    for site_key, config in site_config.items():
        source = Path(
            config["pool_file"]
        )

        if not source.exists():
            raise FileNotFoundError(
                f"Cannot snapshot missing "
                f"{site_key} pool: {source}"
            )

        destination = (
            target
            / f"{site_key}_player_pool.csv"
        )

        shutil.copy2(
            source,
            destination,
        )

        report = inspect_player_pool(
            destination,
            site_key=site_key,
            site_name=config["site_name"],
        )

        sites[site_key] = {
            "site_name": (
                config["site_name"]
            ),
            "source_file": str(source),
            "snapshot_file": (
                destination.name
            ),
            "sha256": sha256_file(
                destination
            ),
            "readiness": report,
        }

    manifest = {
        "schema_version": (
            ARCHIVE_SCHEMA_VERSION
        ),
        "snapshot_id": snapshot_id,
        "captured_at": utc_now_iso(),
        "git_commit": git_commit(),
        "slate": slate,
        "label": label,
        "sites": sites,
    }

    _atomic_json_write(
        target / "manifest.json",
        manifest,
    )

    return manifest


def create_run_archive(
    job_id: str,
    request_data: dict[str, Any],
    site_key: str,
    site_name: str,
    pool_file: Path,
) -> dict[str, Any]:
    run_dir = RUN_ROOT / job_id

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pool_file = Path(pool_file)

    pool_copy = (
        run_dir
        / "player_pool.csv"
    )

    if pool_file.exists():
        shutil.copy2(
            pool_file,
            pool_copy,
        )

    manifest = {
        "schema_version": (
            ARCHIVE_SCHEMA_VERSION
        ),
        "intelligence_version": (
            INTELLIGENCE_VERSION
        ),
        "run_id": job_id,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "completed_at": None,
        "git_commit": git_commit(),
        "site_key": site_key,
        "site_name": site_name,
        "status": "queued",
        "message": "BURN1 run queued.",
        "source_pool": str(pool_file),
        "archived_pool": (
            str(pool_copy)
            if pool_copy.exists()
            else None
        ),
        "pool_sha256": (
            sha256_file(pool_copy)
            if pool_copy.exists()
            else None
        ),
    }

    _atomic_json_write(
        run_dir / "request.json",
        request_data,
    )

    _atomic_json_write(
        run_dir / "manifest.json",
        manifest,
    )

    return manifest


def update_run_archive(
    job_id: str,
    status: str | None = None,
    message: str | None = None,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> bool:
    run_dir = RUN_ROOT / job_id

    manifest_path = (
        run_dir / "manifest.json"
    )

    manifest = _read_json(
        manifest_path
    )

    if not manifest:
        return False

    manifest["updated_at"] = (
        utc_now_iso()
    )

    if status is not None:
        manifest["status"] = status

    if message is not None:
        manifest["message"] = message

    if result is not None:
        _atomic_json_write(
            run_dir / "result.json",
            result,
        )

    if error is not None:
        _atomic_json_write(
            run_dir / "error.json",
            error,
        )

    if status in {
        "complete",
        "error",
    }:
        manifest["completed_at"] = (
            utc_now_iso()
        )

    _atomic_json_write(
        manifest_path,
        manifest,
    )

    return True


def list_run_archives(
    limit: int = 20,
) -> list[dict[str, Any]]:
    if not RUN_ROOT.exists():
        return []

    entries = []

    for run_dir in RUN_ROOT.iterdir():
        if not run_dir.is_dir():
            continue

        manifest = _read_json(
            run_dir / "manifest.json"
        )

        if manifest:
            entries.append(
                manifest
            )

    entries.sort(
        key=lambda item: item.get(
            "created_at",
            "",
        ),
        reverse=True,
    )

    return entries[:limit]
