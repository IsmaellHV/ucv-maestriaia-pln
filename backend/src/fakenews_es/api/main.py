"""Aplicación FastAPI para inferencia del modelo final."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..inference import FinalModelInference, ModelNotLoadedError
from .dependencies import get_inference_engine
from .schemas import HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse

LOGGER = logging.getLogger(__name__)
SCORE_SEMANTICS = (
    "Confianza softmax del clasificador para la clase predicha; no representa una "
    "probabilidad objetiva o factual de que la noticia sea verdadera o falsa."
)


def _cors_origins_from_environment() -> list[str]:
    raw = os.environ.get("FAKENEWS_CORS_ORIGINS", "").strip()
    if not raw:
        return []
    origins = [value.strip() for value in raw.split(",") if value.strip()]
    if "*" in origins:
        raise ValueError("FAKENEWS_CORS_ORIGINS no admite '*' ; declare orígenes explícitos")
    return origins


def create_app(
    engine_factory: Callable[[], FinalModelInference] | None = None,
) -> FastAPI:
    factory = engine_factory or FinalModelInference.from_environment

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        engine = factory()
        await run_in_threadpool(engine.load)
        application.state.inference_engine = engine
        LOGGER.info(
            "Modelo final cargado una vez: version=%s device=%s startup_seconds=%.6f",
            engine.model_info()["model_version"],
            engine.device_name,
            engine.load_seconds,
        )
        yield

    application = FastAPI(
        title="BETO Fake News ES API",
        version="1.0.0",
        description=(
            "Clasificador académico de patrones textuales en noticias en español. "
            "No realiza fact-checking ni consulta fuentes externas. " + SCORE_SEMANTICS
        ),
        lifespan=lifespan,
        redoc_url=None,
    )
    origins = _cors_origins_from_environment()
    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    @application.exception_handler(ModelNotLoadedError)
    async def model_unavailable(_request: Request, exc: ModelNotLoadedError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @application.get("/health", response_model=HealthResponse, tags=["service"])
    def health(request: Request) -> HealthResponse:
        engine = getattr(request.app.state, "inference_engine", None)
        return HealthResponse(status="ok", model_loaded=bool(engine and engine.loaded))

    @application.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
    def model_info(
        engine: FinalModelInference = Depends(get_inference_engine),
    ) -> ModelInfoResponse:
        payload = engine.model_info()
        payload["score_semantics"] = SCORE_SEMANTICS
        return ModelInfoResponse(**payload)

    @application.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    def predict(
        payload: PredictionRequest,
        engine: FinalModelInference = Depends(get_inference_engine),
    ) -> PredictionResponse:
        try:
            result = engine.predict(payload.headline, payload.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return PredictionResponse(**result.to_dict())

    return application


app = create_app()
