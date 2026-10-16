"""Schemas públicos de la API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str = Field(default="", max_length=1000)
    text: str = Field(min_length=1, max_length=100_000)

    @field_validator("text")
    @classmethod
    def text_must_contain_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text no puede estar vacío ni contener sólo espacios")
        return value


class PredictionResponse(BaseModel):
    prediction: str
    label: int = Field(ge=0, le=1)
    score: float = Field(
        ge=0,
        le=1,
        description="Confianza softmax para la clase predicha; no es probabilidad factual.",
    )
    probability_fake: float = Field(
        ge=0,
        le=1,
        description="Salida softmax para Fake=1; no es probabilidad factual.",
    )
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_family: str
    base_checkpoint: str
    base_revision: str
    max_length: int
    labels: dict[int, str]
    positive_class: str
    model_hash: str
    config_hash: str
    dataset_fingerprint: str
    best_epoch: int
    device: str
    startup_seconds: float
    score_semantics: str
