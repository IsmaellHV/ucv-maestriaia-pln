from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fakenews_es.api.main import create_app
from fakenews_es.api.schemas import PredictionRequest
from fakenews_es.inference import FinalModelInference, PredictionResult


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, dict[str, object]]] = []

    def __call__(self, first, second, **kwargs):
        self.calls.append((first, second, kwargs))
        rows = len(first)
        max_length = int(kwargs["max_length"])
        return {
            "input_ids": torch.ones((rows, max_length), dtype=torch.long),
            "attention_mask": torch.ones((rows, max_length), dtype=torch.long),
        }


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(id2label={0: "True", 1: "Fake"})
        self.training = True
        self.eval_calls = 0
        self.device = None
        self.weight = torch.tensor([42.0])

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        self.eval_calls += 1
        return self

    def __call__(self, input_ids, attention_mask):
        del attention_mask
        logits = torch.tensor([[0.0, 2.0]] * len(input_ids), device=input_ids.device)
        return SimpleNamespace(logits=logits)


class InferenceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        model_file = checkpoint / "model.safetensors"
        model_file.write_bytes(b"frozen-weights")
        digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
        metadata = {
            "model_name": "BETO Fake News ES",
            "model_version": "test-version",
            "model_family": "BETO",
            "base_checkpoint": "dccuchile/bert-base-spanish-wwm-cased",
            "base_revision": "revision",
            "canonical_checkpoint": "checkpoint",
            "checkpoint_model_sha256": digest,
            "config_hash": "config-hash",
            "dataset_fingerprint": "fingerprint",
            "best_epoch": 3,
            "max_length": 512,
            "labels": {"0": "True", "1": "Fake"},
            "positive_class": "Fake",
            "input_fields": ["Headline", "Text"],
            "tokenization": {
                "interface": "sequence_pair",
                "truncation": "only_second",
                "padding": "max_length",
                "max_length": 512,
            },
        }
        (self.root / "model.json").write_text(json.dumps(metadata), encoding="utf-8")
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()
        self.tokenizer_loads = 0
        self.model_loads = 0

        def tokenizer_loader(*_args, **_kwargs):
            self.tokenizer_loads += 1
            return self.tokenizer

        def model_loader(*_args, **_kwargs):
            self.model_loads += 1
            return self.model

        self.engine = FinalModelInference(
            project_root=self.root,
            metadata_path=Path("model.json"),
            device="cpu",
            tokenizer_loader=tokenizer_loader,
            model_loader=model_loader,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_model_load_is_idempotent_eval_and_labels_are_frozen(self) -> None:
        self.engine.load()
        self.engine.load()
        self.assertTrue(self.engine.loaded)
        self.assertEqual(self.engine.load_count, 1)
        self.assertEqual(self.tokenizer_loads, 1)
        self.assertEqual(self.model_loads, 1)
        self.assertFalse(self.model.training)
        self.assertEqual(self.model.eval_calls, 1)
        self.assertEqual(self.engine.model_info()["labels"], {0: "True", 1: "Fake"})
        self.assertEqual(self.engine.model_info()["positive_class"], "Fake")

    def test_auto_device_falls_back_to_cpu(self) -> None:
        self.engine.requested_device = "auto"
        with patch("fakenews_es.inference.torch.cuda.is_available", return_value=False):
            self.engine.load()
        self.assertEqual(self.engine.device_name, "cpu")

    def test_prediction_uses_pair_contract_and_does_not_modify_weights(self) -> None:
        self.engine.load()
        before = self.model.weight.clone()
        result = self.engine.predict("", "  Cuerpo obligatorio  ")
        self.assertEqual(result.prediction, "Fake")
        self.assertEqual(result.label, 1)
        self.assertGreaterEqual(result.score, 0)
        self.assertLessEqual(result.score, 1)
        self.assertAlmostEqual(result.score, result.probability_fake)
        first, second, options = self.tokenizer.calls[-1]
        self.assertEqual(first, [""])
        self.assertEqual(second, ["Cuerpo obligatorio"])
        self.assertEqual(options["truncation"], "only_second")
        self.assertEqual(options["padding"], "max_length")
        self.assertEqual(options["max_length"], 512)
        self.assertTrue(torch.equal(before, self.model.weight))

    def test_request_allows_empty_headline_rejects_blank_text_and_metadata(self) -> None:
        payload = PredictionRequest(headline="", text="contenido")
        self.assertEqual(payload.headline, "")
        with self.assertRaises(ValidationError):
            PredictionRequest(headline="x", text="   ")
        with self.assertRaises(ValidationError):
            PredictionRequest(headline="x", text="contenido", Source="forbidden")


class FakeApiEngine:
    def __init__(self) -> None:
        self.loaded = False
        self.load_count = 0
        self.device_name = "cpu"
        self.load_seconds = 0.01

    def load(self):
        self.loaded = True
        self.load_count += 1
        return self

    def model_info(self):
        return {
            "model_name": "BETO Fake News ES",
            "model_version": "test-version",
            "model_family": "BETO",
            "base_checkpoint": "dccuchile/bert-base-spanish-wwm-cased",
            "base_revision": "revision",
            "max_length": 512,
            "labels": {0: "True", 1: "Fake"},
            "positive_class": "Fake",
            "model_hash": "hash",
            "config_hash": "config",
            "dataset_fingerprint": "fingerprint",
            "best_epoch": 3,
            "device": "cpu",
            "startup_seconds": 0.01,
        }

    def predict(self, headline: str, text: str):
        if not text.strip():
            raise ValueError("text vacío")
        return PredictionResult("True", 0, 0.8, 0.2, "test-version")


class ApiContractTests(unittest.TestCase):
    def test_wildcard_cors_is_rejected(self) -> None:
        with patch.dict("os.environ", {"FAKENEWS_CORS_ORIGINS": "*"}, clear=True):
            with self.assertRaises(ValueError):
                create_app(lambda: FakeApiEngine())

    def test_endpoints_openapi_and_single_startup_load(self) -> None:
        engine = FakeApiEngine()
        with patch.dict("os.environ", {}, clear=True):
            app = create_app(lambda: engine)
        with TestClient(app) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok", "model_loaded": True})
            info = client.get("/model-info")
            self.assertEqual(info.status_code, 200)
            self.assertEqual(info.json()["labels"], {"0": "True", "1": "Fake"})
            response = client.post("/predict", json={"headline": "", "text": "contenido"})
            self.assertEqual(response.status_code, 200)
            self.assertIsInstance(response.json()["prediction"], str)
            self.assertIsInstance(response.json()["label"], int)
            self.assertIsInstance(response.json()["score"], float)
            self.assertEqual(
                response.json(),
                {
                    "prediction": "True",
                    "label": 0,
                    "score": 0.8,
                    "probability_fake": 0.2,
                    "model_version": "test-version",
                },
            )
            self.assertEqual(client.get("/docs").status_code, 200)
            self.assertEqual(client.get("/redoc").status_code, 404)
            openapi = client.get("/openapi.json")
            self.assertEqual(openapi.status_code, 200)
            self.assertEqual(set(openapi.json()["paths"]), {"/health", "/model-info", "/predict"})
            self.assertEqual(
                client.post("/predict", json={"headline": "x", "text": "   "}).status_code,
                422,
            )
            self.assertEqual(
                client.post(
                    "/predict",
                    json={"headline": "x", "text": "contenido", "Topic": "forbidden"},
                ).status_code,
                422,
            )
            client.post("/predict", json={"headline": "otro", "text": "contenido"})
        self.assertEqual(engine.load_count, 1)


if __name__ == "__main__":
    unittest.main()
