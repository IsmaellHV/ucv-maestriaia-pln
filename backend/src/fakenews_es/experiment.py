"""Utilidades mínimas para registrar experimentos inmutables."""

from __future__ import annotations

import json
import csv
import hashlib
import os
import platform
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import openpyxl
import pandas as pd
import scipy
import sklearn

REGISTRY_FIELDS = (
    "experiment_id",
    "timestamp",
    "model_family",
    "model_checkpoint",
    "dataset_fingerprint",
    "config_hash",
    "seed",
    "phase",
    "status",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "best_epoch",
    "runtime_seconds",
    "test_accessed_for_model_selection",
    "accessed_for_final_evaluation",
    "notes",
)
VOLATILE_CONFIG_KEYS = {
    "experiment_id",
    "timestamp",
    "timestamp_utc",
    "runtime_seconds",
    "phase",
    "status",
    "config_hash",
    "device",
    "test_accessed_for_model_selection",
    "accessed_for_final_evaluation",
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def utc_now() -> datetime:
    return datetime.now(UTC)


def default_experiment_id(seed: int, timestamp: datetime | None = None) -> str:
    current = timestamp or utc_now()
    return f"{current:%Y%m%d-%H%M}_tfidf-logreg_dev_s{seed}"


def create_run_directory(output_root: Path, experiment_id: str) -> Path:
    run_dir = output_root / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def environment_record() -> dict[str, Any]:
    executable = Path(sys.executable).resolve()
    try:
        executable_record = executable.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        executable_record = executable.as_posix()
    return {
        "python": sys.version.split()[0],
        "python_executable": executable_record,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "openpyxl": openpyxl.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _without_volatile_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _without_volatile_fields(item)
            for key, item in value.items()
            if key not in VOLATILE_CONFIG_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_without_volatile_fields(item) for item in value]
    return value


def config_hash(config: dict[str, Any]) -> str:
    """Hash SHA-256 de JSON canónico, sin identidad ni tiempos de ejecución."""

    effective = _without_volatile_fields(config)
    canonical = json.dumps(
        effective,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dataset_fingerprint(train: Any, validation: Any, policy: dict[str, Any]) -> str:
    payload = {
        "train_sha256": train.source_sha256,
        "development_sha256": validation.source_sha256,
        "train_removed_ids": list(train.removed_ids),
        "development_removed_ids": list(validation.removed_ids),
        "policy": policy,
    }
    # Compatibilidad contractual con las ejecuciones D-012/D-013 existentes.
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def append_experiment_registry(path: Path, row: dict[str, Any]) -> None:
    """Agrega una ejecución sin reemplazar filas históricas."""

    missing = [field for field in REGISTRY_FIELDS if field not in row]
    extras = [field for field in row if field not in REGISTRY_FIELDS]
    if missing or extras:
        raise ValueError(f"Fila de registry inválida: faltan={missing}, extras={extras}")
    existing_ids: set[str] = set()
    if path.is_file():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing_ids = {record["experiment_id"] for record in csv.DictReader(handle)}
    if str(row["experiment_id"]) in existing_ids:
        raise FileExistsError(f"El experimento ya existe en el registry: {row['experiment_id']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row[field] for field in REGISTRY_FIELDS})
