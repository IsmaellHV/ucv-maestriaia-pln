"""Dependencias FastAPI sin lógica de inferencia duplicada."""

from __future__ import annotations

from fastapi import Request

from ..inference import FinalModelInference, ModelNotLoadedError


def get_inference_engine(request: Request) -> FinalModelInference:
    engine = getattr(request.app.state, "inference_engine", None)
    if engine is None or not engine.loaded:
        raise ModelNotLoadedError("El modelo no está disponible")
    return engine
