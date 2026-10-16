"""Inferencia del modelo BETO final congelado, independiente de la capa HTTP."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .beto import split_input_pair
from .data import FAKE_LABEL, TRUE_LABEL, build_input_text, file_sha256


class ModelIntegrityError(RuntimeError):
    """La metadata o el checkpoint no coincide con el modelo final congelado."""


class ModelNotLoadedError(RuntimeError):
    """Se intentó inferir antes de completar la carga del modelo."""


@dataclass(frozen=True)
class PredictionResult:
    prediction: str
    label: int
    score: float
    probability_fake: float
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class FinalModelInference:
    """Carga una vez el checkpoint canónico y reutiliza la misma instancia para inferencia."""

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        metadata_path: Path | None = None,
        device: str = "auto",
        tokenizer_loader: Callable[..., Any] | None = None,
        model_loader: Callable[..., Any] | None = None,
    ) -> None:
        self.project_root = (project_root or default_project_root()).resolve()
        requested_metadata = metadata_path or Path("results/final_model.json")
        self.metadata_path = (
            requested_metadata
            if requested_metadata.is_absolute()
            else self.project_root / requested_metadata
        ).resolve()
        self.requested_device = device.lower()
        self._tokenizer_loader = tokenizer_loader or AutoTokenizer.from_pretrained
        self._model_loader = model_loader or AutoModelForSequenceClassification.from_pretrained
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._metadata: dict[str, Any] | None = None
        self._checkpoint: Path | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: torch.device | None = None
        self._load_seconds: float | None = None
        self._load_count = 0

    @classmethod
    def from_environment(cls, *, project_root: Path | None = None) -> "FinalModelInference":
        metadata = Path(os.environ.get("FAKENEWS_MODEL_METADATA", "results/final_model.json"))
        device = os.environ.get("FAKENEWS_DEVICE", "auto")
        return cls(project_root=project_root, metadata_path=metadata, device=device)

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def load_count(self) -> int:
        return self._load_count

    @property
    def device_name(self) -> str:
        return str(self._device) if self._device is not None else "not_loaded"

    @property
    def load_seconds(self) -> float | None:
        return self._load_seconds

    @property
    def model(self) -> Any:
        if self._model is None:
            raise ModelNotLoadedError("El modelo final todavía no está cargado")
        return self._model

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise ModelNotLoadedError("El tokenizer final todavía no está cargado")
        return self._tokenizer

    def _load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"No existe metadata del modelo final: {self.metadata_path}")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        expected = {
            "model_family": "BETO",
            "max_length": 512,
            "positive_class": "Fake",
            "best_epoch": 3,
        }
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise ModelIntegrityError(
                    f"Metadata final inválida: {field}={metadata.get(field)!r}; esperado={value!r}"
                )
        labels = {int(key): value for key, value in metadata.get("labels", {}).items()}
        if labels != {TRUE_LABEL: "True", FAKE_LABEL: "Fake"}:
            raise ModelIntegrityError(f"Etiquetas finales inválidas: {labels}")
        tokenization = metadata.get("tokenization", {})
        expected_tokenization = {
            "interface": "sequence_pair",
            "truncation": "only_second",
            "padding": "max_length",
            "max_length": 512,
        }
        if tokenization != expected_tokenization:
            raise ModelIntegrityError(f"Tokenización final inválida: {tokenization}")
        if metadata.get("input_fields") != ["Headline", "Text"]:
            raise ModelIntegrityError("El modelo final debe usar sólo Headline y Text")
        return metadata

    def _resolve_checkpoint(self, metadata: dict[str, Any]) -> Path:
        raw = Path(metadata["canonical_checkpoint"])
        checkpoint = (raw if raw.is_absolute() else self.project_root / raw).resolve()
        try:
            checkpoint.relative_to(self.project_root)
        except ValueError as exc:
            raise ModelIntegrityError("El checkpoint final debe estar dentro del proyecto") from exc
        model_file = checkpoint / "model.safetensors"
        if not model_file.is_file():
            raise FileNotFoundError(f"No existe el modelo final: {model_file}")
        actual_hash = file_sha256(model_file)
        expected_hash = metadata["checkpoint_model_sha256"]
        if actual_hash != expected_hash:
            raise ModelIntegrityError(
                f"Hash del modelo final inválido: {actual_hash}; esperado={expected_hash}"
            )
        return checkpoint

    def _resolve_device(self) -> torch.device:
        if self.requested_device == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.requested_device == "cpu":
            return torch.device("cpu")
        if self.requested_device in {"cuda", "cuda:0"}:
            if not torch.cuda.is_available():
                raise RuntimeError("Se solicitó CUDA, pero no está disponible")
            return torch.device("cuda:0")
        raise ValueError("FAKENEWS_DEVICE debe ser auto, cpu, cuda o cuda:0")

    def load(self) -> "FinalModelInference":
        with self._load_lock:
            if self.loaded:
                return self
            started = time.perf_counter()
            metadata = self._load_metadata()
            checkpoint = self._resolve_checkpoint(metadata)
            device = self._resolve_device()
            tokenizer = self._tokenizer_loader(checkpoint, local_files_only=True)
            model = self._model_loader(checkpoint, local_files_only=True)
            checkpoint_labels = {
                int(key): value for key, value in dict(model.config.id2label).items()
            }
            if checkpoint_labels != {TRUE_LABEL: "True", FAKE_LABEL: "Fake"}:
                raise ModelIntegrityError(
                    f"Etiquetas del checkpoint incompatibles: {checkpoint_labels}"
                )
            model.to(device)
            model.eval()
            self._metadata = metadata
            self._checkpoint = checkpoint
            self._tokenizer = tokenizer
            self._model = model
            self._device = device
            self._load_seconds = time.perf_counter() - started
            self._load_count += 1
            return self

    def model_info(self) -> dict[str, Any]:
        if not self.loaded or self._metadata is None:
            raise ModelNotLoadedError("El modelo final todavía no está cargado")
        metadata = self._metadata
        return {
            "model_name": metadata["model_name"],
            "model_version": metadata["model_version"],
            "model_family": metadata["model_family"],
            "base_checkpoint": metadata["base_checkpoint"],
            "base_revision": metadata["base_revision"],
            "max_length": metadata["max_length"],
            "labels": {int(key): value for key, value in metadata["labels"].items()},
            "positive_class": metadata["positive_class"],
            "model_hash": metadata["checkpoint_model_sha256"],
            "config_hash": metadata["config_hash"],
            "dataset_fingerprint": metadata["dataset_fingerprint"],
            "best_epoch": metadata["best_epoch"],
            "device": self.device_name,
            "startup_seconds": self._load_seconds,
        }

    def predict(self, headline: str, text: str) -> PredictionResult:
        return self.predict_many([(headline, text)], batch_size=1)[0]

    def predict_many(
        self, records: Sequence[tuple[str, str]], *, batch_size: int = 4
    ) -> list[PredictionResult]:
        if not self.loaded or self._metadata is None or self._device is None:
            raise ModelNotLoadedError("El modelo final todavía no está cargado")
        if not records:
            return []
        if batch_size < 1:
            raise ValueError("batch_size debe ser positivo")
        pairs = [split_input_pair(build_input_text(headline, text)) for headline, text in records]
        results: list[PredictionResult] = []
        with self._inference_lock, torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                encoded = self.tokenizer(
                    [pair.headline for pair in batch],
                    [pair.text for pair in batch],
                    truncation="only_second",
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt",
                )
                inputs = {key: value.to(self._device) for key, value in encoded.items()}
                logits = self.model(**inputs).logits
                probabilities = torch.softmax(logits.float(), dim=-1).cpu()
                labels = torch.argmax(probabilities, dim=-1).tolist()
                for index, label in enumerate(labels):
                    probability_fake = float(probabilities[index, FAKE_LABEL])
                    score = float(probabilities[index, label])
                    results.append(
                        PredictionResult(
                            prediction=self._metadata["labels"][str(label)],
                            label=int(label),
                            score=score,
                            probability_fake=probability_fake,
                            model_version=self._metadata["model_version"],
                        )
                    )
        return results
