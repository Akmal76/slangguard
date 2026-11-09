"""Prediction API.

    uv run uvicorn src.serving.app:create_default_app --factory

GET /health tells whether the model is loaded, POST /predict labels texts. The interactive
documentation is at /docs.
"""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.data.clean import normalize_text
from src.serving.config import load_serving_config
from src.serving.metrics import Metrics
from src.serving.model import SentimentModel


class PredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, description="The texts to label, in any order.")


class Prediction(BaseModel):
    label: str
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    predictions: list[Prediction]  # in the same order as the request
    model: dict[str, str]


def check_texts(texts: list[str], config: dict) -> None:
    """Reject requests that are too big, or that the model could not read."""
    if len(texts) > config["max_texts"]:
        raise HTTPException(422, f"At most {config['max_texts']} texts per request")
    for i, text in enumerate(texts):
        if len(text) > config["max_chars"]:
            raise HTTPException(422, f"Text {i} is longer than {config['max_chars']} characters")
        # Training never saw empty text: cleaning dropped it. So it is refused, not guessed.
        if not normalize_text(text):
            raise HTTPException(422, f"Text {i} has no words in it")


def create_app(config: dict, model: SentimentModel | None = None) -> FastAPI:
    """Build the API. Without a model, the one named in the config is loaded at start."""

    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = model or SentimentModel.load(config)
        info = app.state.model.info
        metrics.model_info.labels(info["adapter_repo"], info["adapter_revision"]).set(1)
        yield

    app = FastAPI(title="SlangGuard", lifespan=lifespan)

    @app.middleware("http")
    async def record_request(request: Request, call_next):
        start = time.perf_counter()
        status = "500"  # what is recorded if the handler crashes
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            # The route template ("/predict"), not the raw path, so random URLs cannot create
            # an endless number of time series.
            route = request.scope.get("route")
            endpoint = route.path if route else "unmatched"
            metrics.requests.labels(endpoint, status).inc()
            metrics.request_seconds.labels(endpoint).observe(time.perf_counter() - start)

    @app.get("/metrics", include_in_schema=False)
    def read_metrics():
        body, content_type = metrics.render()
        return Response(body, media_type=content_type)

    @app.get("/health")
    def health(request: Request):
        return {"status": "ok", "model": request.app.state.model.info}

    # A plain `def`, not `async def`: FastAPI runs it in a worker thread, so a slow
    # prediction does not block the server.
    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest, request: Request):
        check_texts(body.texts, config)
        sentiment = request.app.state.model
        results = sentiment.predict_detailed(body.texts)
        metrics.record_predictions(results)
        predictions = [
            Prediction(
                label=max(r.probabilities, key=r.probabilities.get), probabilities=r.probabilities
            )
            for r in results
        ]
        return PredictResponse(predictions=predictions, model=sentiment.info)

    return app


def create_default_app() -> FastAPI:
    path = Path(os.environ.get("SERVING_CONFIG", "configs/serving.yaml"))
    return create_app(load_serving_config(path, os.environ))
