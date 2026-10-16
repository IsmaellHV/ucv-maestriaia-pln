"""Carga, validación y preparación reproducible de los datasets."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

TRUE_LABEL = 0
FAKE_LABEL = 1
FEATURE_FIELDS = ("Headline", "Text")
EXCLUDED_FEATURE_FIELDS = ("Id", "Source", "Link", "Topic")
TEXT_SEPARATOR = "\n\n"
FINAL_TEST_ACCESS_TOKEN = "FINAL_EVALUATION_ONLY"
D012_NEAR_COSINE_THRESHOLD = 0.90
D012_NEAR_LENGTH_RATIO_THRESHOLD = 0.80
D012_NEAR_SEQUENCE_THRESHOLD = 0.85

EXPECTED_COLUMNS = ("Id", "Category", "Topic", "Source", "Headline", "Text", "Link")
EXPECTED_RAW_ROWS = {"train": 676, "development": 295, "test": 572}
FILE_NAMES = {"train": "train.xlsx", "development": "development.xlsx", "test": "test.xlsx"}
ALIASES = {
    "id": "Id",
    "category": "Category",
    "topic": "Topic",
    "topics": "Topic",
    "source": "Source",
    "headline": "Headline",
    "text": "Text",
    "link": "Link",
}


class DatasetValidationError(ValueError):
    """Indica que un dataset viola el contrato experimental."""


class TestSetQuarantinedError(PermissionError):
    """Impide cargar test de forma accidental durante selección."""


@dataclass(frozen=True)
class PreparedSplit:
    name: str
    X: pd.Series
    y: pd.Series
    metadata: pd.DataFrame
    audit_frame: pd.DataFrame
    raw_rows: int
    removed_ids: tuple[int, ...]
    source_path: Path
    source_sha256: str
    feature_fields: tuple[str, ...] = FEATURE_FIELDS


@dataclass(frozen=True)
class ExperimentalSplits:
    """Particiones derivadas conjuntamente y evidencia de D-012."""

    train: PreparedSplit
    development: PreparedSplit
    exclusion_manifest: tuple[dict[str, Any], ...]
    policy: dict[str, Any]


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"\s+", " ", normalized)


def map_label(value: Any) -> int:
    if pd.isna(value):
        raise DatasetValidationError("La etiqueta no puede estar vacía")
    if isinstance(value, (bool, np.bool_)):
        return TRUE_LABEL if bool(value) else FAKE_LABEL
    normalized = normalize_text(value)
    if normalized in {"true", "verdadera", "real"}:
        return TRUE_LABEL
    if normalized in {"fake", "falsa", "false"}:
        return FAKE_LABEL
    raise DatasetValidationError(f"Etiqueta no reconocida: {value!r}")


def build_input_text(headline: Any, text: Any) -> str:
    headline_value = "" if pd.isna(headline) else str(headline).strip()
    text_value = "" if pd.isna(text) else str(text).strip()
    if not text_value:
        raise DatasetValidationError("Text no puede estar vacío")
    return f"{headline_value}{TEXT_SEPARATOR}{text_value}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_text_sha256(value: Any) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def _canonicalize_columns(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    renamed = frame.rename(columns={column: ALIASES.get(normalize_text(column), column) for column in frame.columns})
    missing = [column for column in EXPECTED_COLUMNS if column not in renamed.columns]
    extras = [column for column in renamed.columns if column not in EXPECTED_COLUMNS]
    if missing or extras:
        raise DatasetValidationError(f"Esquema inválido en {source}: faltan={missing}, extras={extras}")
    return renamed.loc[:, EXPECTED_COLUMNS].copy()


def _clean_development(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[int, ...]]:
    ids = set(frame["Id"].tolist())
    if 129 not in ids or 147 not in ids:
        raise DatasetValidationError("Development debe contener Id=129 e Id=147 antes de depurar")
    comparison_fields = ("Category", "Topic", "Source", "Headline", "Text", "Link")
    kept = frame.loc[frame["Id"].eq(129), comparison_fields]
    removed = frame.loc[frame["Id"].eq(147), comparison_fields]
    if len(kept) != 1 or len(removed) != 1:
        raise DatasetValidationError("Id=129 e Id=147 deben ser únicos en development")
    kept_values = tuple(normalize_text(value) for value in kept.iloc[0])
    removed_values = tuple(normalize_text(value) for value in removed.iloc[0])
    if kept_values != removed_values:
        raise DatasetValidationError("Id=147 ya no coincide con Id=129; se detiene la depuración")
    cleaned = frame.loc[~frame["Id"].eq(147)].reset_index(drop=True)
    if len(cleaned) != 294 or 147 in set(cleaned["Id"]):
        raise DatasetValidationError("Development depurado debe tener 294 filas y excluir Id=147")
    return cleaned, (147,)


def _validate_no_content_duplicates(frame: pd.DataFrame, split: str) -> None:
    keys = frame["Headline"].map(normalize_text) + "\n" + frame["Text"].map(normalize_text)
    duplicated = keys.str.strip().ne("") & keys.duplicated(keep=False)
    if duplicated.any():
        ids = frame.loc[duplicated, "Id"].tolist()
        raise DatasetValidationError(f"Duplicados de contenido inesperados en {split}: Id={ids}")


def _validate_feature_contract(frame: pd.DataFrame, features: pd.Series) -> None:
    expected = frame.apply(lambda row: build_input_text(row["Headline"], row["Text"]), axis=1)
    expected.index = features.index
    if not features.equals(expected):
        raise DatasetValidationError("Las features contienen información distinta de Headline + Text")


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    split: str,
    raw_rows: int,
    removed_ids: tuple[int, ...],
    source_path: Path,
    source_sha256: str,
) -> PreparedSplit:
    _validate_no_content_duplicates(frame, split)
    labels = frame["Category"].map(map_label).astype("int8")
    invalid = set(labels.unique()) - {TRUE_LABEL, FAKE_LABEL}
    if invalid:
        raise DatasetValidationError(f"Etiquetas binarias inválidas en {split}: {sorted(invalid)}")

    features = frame.apply(lambda row: build_input_text(row["Headline"], row["Text"]), axis=1)
    features.name = "input_text"
    _validate_feature_contract(frame, features)
    metadata = frame.loc[:, EXCLUDED_FEATURE_FIELDS].copy()
    if tuple(FEATURE_FIELDS) != ("Headline", "Text") or any(
        field in FEATURE_FIELDS for field in EXCLUDED_FEATURE_FIELDS
    ):
        raise DatasetValidationError("Contrato de features inválido")
    return PreparedSplit(
        name=split,
        X=features.reset_index(drop=True),
        y=labels.reset_index(drop=True),
        metadata=metadata.reset_index(drop=True),
        audit_frame=frame.reset_index(drop=True).copy(),
        raw_rows=raw_rows,
        removed_ids=removed_ids,
        source_path=source_path,
        source_sha256=source_sha256,
    )


def _near_duplicate_candidates(
    train: pd.DataFrame,
    development: pd.DataFrame,
) -> list[dict[str, Any]]:
    train_texts = train["Text"].map(normalize_text).tolist()
    development_texts = development["Text"].map(normalize_text).tolist()
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(5, 5),
        lowercase=False,
        norm="l2",
        dtype=np.float64,
    )
    matrix = vectorizer.fit_transform(train_texts + development_texts)
    similarities = (matrix[: len(train)] @ matrix[len(train) :].T).toarray()
    train_indices, development_indices = np.where(
        similarities >= D012_NEAR_COSINE_THRESHOLD
    )
    candidates: list[dict[str, Any]] = []
    for train_index, development_index in zip(train_indices, development_indices, strict=True):
        left_text = train_texts[int(train_index)]
        right_text = development_texts[int(development_index)]
        if left_text == right_text:
            continue
        length_ratio = min(len(left_text), len(right_text)) / max(len(left_text), len(right_text))
        if length_ratio < D012_NEAR_LENGTH_RATIO_THRESHOLD:
            continue
        sequence_ratio = SequenceMatcher(None, left_text, right_text, autojunk=False).ratio()
        if sequence_ratio < D012_NEAR_SEQUENCE_THRESHOLD:
            continue
        left = train.iloc[int(train_index)]
        right = development.iloc[int(development_index)]
        left_link = normalize_text(left["Link"])
        right_link = normalize_text(right["Link"])
        candidates.append(
            {
                "train_id": int(left["Id"]),
                "development_id": int(right["Id"]),
                "train_normalized_hash": normalized_text_sha256(left["Text"]),
                "development_normalized_hash": normalized_text_sha256(right["Text"]),
                "cosine_similarity": float(similarities[train_index, development_index]),
                "length_ratio": float(length_ratio),
                "sequence_matcher_ratio": float(sequence_ratio),
                "normalized_link": left_link if left_link and left_link == right_link else None,
            }
        )
    return sorted(candidates, key=lambda item: (item["train_id"], item["development_id"]))


def _apply_d012(
    train: pd.DataFrame,
    development: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[dict[str, Any], ...]]:
    train_keys = train["Text"].map(normalize_text)
    development_keys = development["Text"].map(normalize_text)
    common_keys = (set(train_keys) - {""}) & (set(development_keys) - {""})
    exclusions_by_id: dict[int, dict[str, Any]] = {}

    for key in sorted(common_keys):
        counterpart_ids = sorted(
            int(value) for value in development.loc[development_keys.eq(key), "Id"].tolist()
        )
        for _, row in train.loc[train_keys.eq(key)].iterrows():
            train_id = int(row["Id"])
            exclusions_by_id[train_id] = {
                "split": "train",
                "Id": train_id,
                "reason": "El grupo aparece en TRAIN y DEVELOPMENT; DEVELOPMENT tiene precedencia.",
                "match_type": "normalized_text_exact",
                "counterpart_id": counterpart_ids[0],
                "counterpart_ids": counterpart_ids,
                "normalized_hash": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                "counterpart_normalized_hash": hashlib.sha256(key.encode("utf-8")).hexdigest(),
                "Link": None,
                "rule_applied": "D-012(a): Text normalizado idéntico + precedencia de DEVELOPMENT",
            }

    for candidate in _near_duplicate_candidates(train, development):
        if candidate["normalized_link"] is None:
            continue
        train_id = int(candidate["train_id"])
        if train_id in exclusions_by_id:
            continue
        exclusions_by_id[train_id] = {
            "split": "train",
            "Id": train_id,
            "reason": "Near-duplicate corroborado por Link; DEVELOPMENT tiene precedencia.",
            "match_type": "near_duplicate_same_normalized_link",
            "counterpart_id": int(candidate["development_id"]),
            "counterpart_ids": [int(candidate["development_id"])],
            "normalized_hash": candidate["train_normalized_hash"],
            "counterpart_normalized_hash": candidate["development_normalized_hash"],
            "Link": candidate["normalized_link"],
            "rule_applied": (
                "D-012(b): cosine>=0.90 + length_ratio>=0.80 + "
                "SequenceMatcher>=0.85 + Link normalizado compartido + precedencia de DEVELOPMENT"
            ),
            "cosine_similarity": candidate["cosine_similarity"],
            "length_ratio": candidate["length_ratio"],
            "sequence_matcher_ratio": candidate["sequence_matcher_ratio"],
        }

    exclusion_manifest = tuple(exclusions_by_id[key] for key in sorted(exclusions_by_id))
    removed_ids = {record["Id"] for record in exclusion_manifest}
    filtered = train.loc[~train["Id"].isin(removed_ids)].reset_index(drop=True)
    return filtered, exclusion_manifest


def _validate_d012_separation(train: pd.DataFrame, development: pd.DataFrame) -> None:
    common = (set(train["Text"].map(normalize_text)) - {""}) & (
        set(development["Text"].map(normalize_text)) - {""}
    )
    if common:
        raise DatasetValidationError("D-012 falló: persisten Text normalizados idénticos")
    violating_near = [
        candidate
        for candidate in _near_duplicate_candidates(train, development)
        if candidate["normalized_link"] is not None
    ]
    if violating_near:
        pairs = [
            (candidate["train_id"], candidate["development_id"])
            for candidate in violating_near
        ]
        raise DatasetValidationError(f"D-012 falló: persisten near-duplicates con Link: {pairs}")


def load_split(
    dataset_dir: Path | str,
    split: str,
    *,
    test_access_token: str | None = None,
) -> PreparedSplit:
    split = split.lower()
    if split not in FILE_NAMES:
        raise DatasetValidationError(f"Partición desconocida: {split}")
    if split == "test" and test_access_token != FINAL_TEST_ACCESS_TOKEN:
        raise TestSetQuarantinedError(
            "TEST está en cuarentena. Sólo puede cargarse para la evaluación final aprobada "
            f"con el token explícito {FINAL_TEST_ACCESS_TOKEN!r}."
        )

    source = Path(dataset_dir) / FILE_NAMES[split]
    if not source.is_file():
        raise FileNotFoundError(f"No existe el dataset: {source}")
    frame = _canonicalize_columns(pd.read_excel(source), source)
    raw_rows = len(frame)
    if raw_rows != EXPECTED_RAW_ROWS[split]:
        raise DatasetValidationError(
            f"{split} debe tener {EXPECTED_RAW_ROWS[split]} filas originales; tiene {raw_rows}"
        )

    removed_ids: tuple[int, ...] = ()
    if split == "development":
        frame, removed_ids = _clean_development(frame)
    return _prepare_frame(
        frame,
        split=split,
        raw_rows=raw_rows,
        removed_ids=removed_ids,
        source_path=source,
        source_sha256=file_sha256(source),
    )


def load_experimental_splits(dataset_dir: Path | str) -> ExperimentalSplits:
    """Deriva TRAIN/DEVELOPMENT con D-005 y D-012, sin abrir TEST."""

    dataset_dir = Path(dataset_dir)
    train_source = dataset_dir / FILE_NAMES["train"]
    development_source = dataset_dir / FILE_NAMES["development"]
    train_raw = _canonicalize_columns(pd.read_excel(train_source), train_source)
    development_raw = _canonicalize_columns(pd.read_excel(development_source), development_source)
    if len(train_raw) != EXPECTED_RAW_ROWS["train"]:
        raise DatasetValidationError("TRAIN original debe contener 676 filas")
    if len(development_raw) != EXPECTED_RAW_ROWS["development"]:
        raise DatasetValidationError("DEVELOPMENT original debe contener 295 filas")

    development_frame, development_removed = _clean_development(development_raw)
    train_frame, exclusion_manifest = _apply_d012(train_raw, development_frame)
    train_removed = tuple(int(record["Id"]) for record in exclusion_manifest)
    if len(train_frame) != 671 or len(development_frame) != 294:
        raise DatasetValidationError(
            f"D-012 produjo tamaños inesperados: train={len(train_frame)}, development={len(development_frame)}"
        )
    _validate_d012_separation(train_frame, development_frame)

    train = _prepare_frame(
        train_frame,
        split="train_experimental",
        raw_rows=len(train_raw),
        removed_ids=train_removed,
        source_path=train_source,
        source_sha256=file_sha256(train_source),
    )
    development = _prepare_frame(
        development_frame,
        split="development_experimental",
        raw_rows=len(development_raw),
        removed_ids=development_removed,
        source_path=development_source,
        source_sha256=file_sha256(development_source),
    )
    policy = {
        "decision": "D-012",
        "status": "accepted",
        "development_precedence": True,
        "normalization": "Unicode NFKC + lowercase + trim + collapsed whitespace",
        "near_duplicate": {
            "analyzer": "char_wb",
            "ngram_range": [5, 5],
            "cosine_threshold": D012_NEAR_COSINE_THRESHOLD,
            "length_ratio_threshold": D012_NEAR_LENGTH_RATIO_THRESHOLD,
            "sequence_matcher_threshold": D012_NEAR_SEQUENCE_THRESHOLD,
            "requires_shared_nonempty_normalized_link": True,
        },
    }
    return ExperimentalSplits(
        train=train,
        development=development,
        exclusion_manifest=exclusion_manifest,
        policy=policy,
    )
