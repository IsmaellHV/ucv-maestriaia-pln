"""Primer experimento BETO, con tokenización por pares y TEST en cuarentena."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import psutil
import torch
import accelerate
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .data import EXCLUDED_FEATURE_FIELDS, FEATURE_FIELDS, TEXT_SEPARATOR, load_experimental_splits
from .evaluation import evaluate_binary
from .experiment import (
    append_experiment_registry,
    config_hash,
    create_run_directory,
    dataset_fingerprint,
    environment_record,
    utc_now,
    write_json,
)


@dataclass(frozen=True)
class InputPair:
    headline: str
    text: str


def split_input_pair(value: str) -> InputPair:
    headline, separator, text = value.partition(TEXT_SEPARATOR)
    if not separator or not text:
        raise ValueError("La entrada debe conservar Headline + separador + Text")
    return InputPair(headline=headline, text=text)


def _summary(values: Sequence[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.int64)
    return {
        "min": int(array.min()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": int(array.max()),
    }


def analyze_token_lengths(
    input_texts: Sequence[str], tokenizer: Any, candidates: Sequence[int] = (128, 256, 512)
) -> dict[str, Any]:
    headline_lengths: list[int] = []
    text_lengths: list[int] = []
    pair_lengths: list[int] = []
    pair_special_tokens = int(tokenizer.num_special_tokens_to_add(pair=True))
    for value in input_texts:
        pair = split_input_pair(value)
        headline_ids = tokenizer(pair.headline, add_special_tokens=False)["input_ids"]
        text_ids = tokenizer(pair.text, add_special_tokens=False)["input_ids"]
        headline_lengths.append(len(headline_ids))
        text_lengths.append(len(text_ids))
        pair_lengths.append(len(headline_ids) + len(text_ids) + pair_special_tokens)

    truncation: dict[str, Any] = {}
    total_pair_tokens = sum(pair_lengths)
    for max_length in candidates:
        discarded: list[int] = []
        for headline_length, text_length in zip(headline_lengths, text_lengths, strict=True):
            available_for_text = max_length - headline_length - pair_special_tokens
            if available_for_text < 0:
                raise ValueError(f"Headline no cabe a max_length={max_length}")
            discarded.append(max(0, text_length - available_for_text))
        total_discarded = sum(discarded)
        truncated_examples = sum(value > 0 for value in discarded)
        truncation[str(max_length)] = {
            "examples_truncated": truncated_examples,
            "examples_truncated_percent": 100.0 * truncated_examples / len(pair_lengths),
            "tokens_discarded": total_discarded,
            "tokens_discarded_percent_of_full_pairs": 100.0 * total_discarded / total_pair_tokens,
            "attention_cost_relative_to_128": (max_length / 128) ** 2,
        }
    return {
        "scope": "train_experimental_only",
        "rows": len(pair_lengths),
        "counting": {
            "headline": "subtokens without special tokens",
            "text": "subtokens without special tokens",
            "pair": "headline and text subtokens plus pair special tokens",
            "pair_special_tokens": pair_special_tokens,
        },
        "headline": _summary(headline_lengths),
        "text": _summary(text_lengths),
        "pair": _summary(pair_lengths),
        "distributions": {
            "headline": headline_lengths,
            "text": text_lengths,
            "pair": pair_lengths,
        },
        "truncation_only_second": truncation,
    }


class EncodedPairDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, input_texts: Sequence[str], labels: Sequence[int], tokenizer: Any, max_length: int):
        pairs = [split_input_pair(value) for value in input_texts]
        encoded = tokenizer(
            [pair.headline for pair in pairs],
            [pair.text for pair in pairs],
            truncation="only_second",
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        self.encoded = dict(encoded)
        self.labels = torch.as_tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encoded.items()}
        item["labels"] = self.labels[index]
        return item


def set_transformer_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def hardware_record() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    record: dict[str, Any] = {
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "ram_total_gib": memory.total / 2**30,
        "ram_available_gib_at_start": memory.available / 2**30,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        record["gpu"] = {
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_gib": total / 2**30,
            "free_memory_gib_at_start": free / 2**30,
            "supported_architectures": torch.cuda.get_arch_list(),
        }
    return record


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if tuple(config["features"]["fields"]) != FEATURE_FIELDS:
        raise ValueError("BETO debe usar Headline y Text")
    if tuple(config["features"]["excluded_fields"]) != EXCLUDED_FEATURE_FIELDS:
        raise ValueError("Los metadatos excluidos no coinciden con el contrato")
    if config["tokenization"]["truncation"] != "only_second":
        raise ValueError("La truncación debe preservar Headline")
    if config["selection"]["positive_label"] != 1:
        raise ValueError("Fake debe ser la clase positiva")
    if config["test_policy"] != "quarantined_not_loaded_or_evaluated":
        raise ValueError("TEST debe permanecer en cuarentena")
    training = config["training"]
    if training["effective_batch_size"] != (
        training["train_micro_batch_size"] * training["gradient_accumulation_steps"]
    ):
        raise ValueError("El batch efectivo declarado no coincide")
    return config


def checkpoint_selection_key(metrics: dict[str, Any], validation_loss: float, epoch: int) -> tuple[float, float, float, int]:
    """F1, recall, menor loss y época más temprana, en ese orden."""

    return (float(metrics["f1"]), float(metrics["recall"]), -float(validation_loss), -int(epoch))


def is_better_checkpoint(
    metrics: dict[str, Any],
    validation_loss: float,
    epoch: int,
    best_key: tuple[float, float, float, int] | None,
) -> bool:
    return best_key is None or checkpoint_selection_key(metrics, validation_loss, epoch) > best_key


def _evaluate(
    model: Any, loader: DataLoader[Any], device: torch.device
) -> tuple[list[int], list[float], float]:
    model.eval()
    predictions: list[int] = []
    fake_probabilities: list[float] = []
    loss_sum = 0.0
    examples = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = {key: value.to(device) for key, value in batch.items()}
            output = model(**inputs)
            logits = output.logits
            batch_size = int(inputs["labels"].shape[0])
            loss_sum += float(output.loss.detach().cpu()) * batch_size
            examples += batch_size
            probabilities = torch.softmax(logits.float(), dim=-1)
            predictions.extend(torch.argmax(probabilities, dim=-1).cpu().tolist())
            fake_probabilities.extend(probabilities[:, 1].cpu().tolist())
    return predictions, fake_probabilities, loss_sum / examples


def run_experiment(
    dataset_dir: Path,
    config_path: Path,
    output_root: Path,
    experiment_id: str | None = None,
    *,
    status: str = "candidate",
    registry_path: Path | None = Path("results/experiment_registry.csv"),
) -> Path:
    started = time.perf_counter()
    timestamp = utc_now()
    config = load_config(config_path)
    if status not in {
        "candidate",
        "selected_development",
        "reproduction",
        "selected_for_final_test",
        "final_test",
    }:
        raise ValueError(f"Estado experimental no reconocido: {status}")
    effective_config_hash = config_hash(config)
    seed = int(config["seed"])
    set_transformer_seed(seed)
    hardware = hardware_record()
    if not hardware["cuda_available"]:
        raise RuntimeError("BETO inicial requiere CUDA en este equipo; CPU no es práctico")

    derived = load_experimental_splits(dataset_dir)
    train, validation = derived.train, derived.development
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["checkpoint"], revision=model_config["revision"]
    )
    lengths = analyze_token_lengths(train.X.tolist(), tokenizer)
    max_length = int(config["tokenization"]["max_length"])
    training = config["training"]
    train_dataset = EncodedPairDataset(train.X.tolist(), train.y.tolist(), tokenizer, max_length)
    validation_dataset = EncodedPairDataset(
        validation.X.tolist(), validation.y.tolist(), tokenizer, max_length
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["train_micro_batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    experiment_id = experiment_id or f"{timestamp:%Y%m%d-%H%M%S}_beto-initial_dev_s{seed}"
    run_dir = create_run_directory(output_root, experiment_id)
    checkpoint_dir = run_dir / "checkpoint-best"
    experimental_fingerprint = dataset_fingerprint(train, validation, derived.policy)
    resolved = {
        **config,
        "experiment_id": experiment_id,
        "timestamp_utc": timestamp.isoformat(),
        "config_hash": effective_config_hash,
        "phase": "development",
        "status": status,
        "device": "cuda:0",
        "test_accessed_for_model_selection": False,
    }
    write_json(run_dir / "config.json", resolved)
    environment = environment_record()
    environment.update(
        {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "accelerate": accelerate.__version__,
            "psutil": psutil.__version__,
        }
    )
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "token_length_analysis.train.json", lengths)
    write_json(
        run_dir / "data_manifest.json",
        {
            "train": {"sha256": train.source_sha256, "raw_rows": train.raw_rows, "used_rows": len(train.X), "removed_ids": list(train.removed_ids), "label_distribution": {"True": int((train.y == 0).sum()), "Fake": int((train.y == 1).sum())}},
            "validation": {"sha256": validation.source_sha256, "raw_rows": validation.raw_rows, "used_rows": len(validation.X), "removed_ids": list(validation.removed_ids), "label_distribution": {"True": int((validation.y == 0).sum()), "Fake": int((validation.y == 1).sum())}},
            "dataset_fingerprint": experimental_fingerprint,
            "deduplication": {"policy": derived.policy, "exclusions": list(derived.exclusion_manifest)},
            "features": list(FEATURE_FIELDS),
            "excluded_predictors": list(EXCLUDED_FEATURE_FIELDS),
            "test": {"status": "not_loaded", "metrics": None, "test_accessed_for_model_selection": False},
        },
    )

    device = torch.device("cuda:0")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config["checkpoint"],
        revision=model_config["revision"],
        num_labels=int(model_config["num_labels"]),
    )
    model.config.id2label = {0: "True", 1: "Fake"}
    model.config.label2id = {"True": 0, "Fake": 1}
    if training["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    accumulation = int(training["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = updates_per_epoch * int(training["epochs"])
    warmup_steps = round(total_updates * float(training["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=training["mixed_precision"] == "fp16")

    epochs: list[dict[str, Any]] = []
    best_key: tuple[float, float, float, int] | None = None
    best_epoch = 0
    no_improvement = 0
    peak_allocated_bytes = 0
    peak_reserved_bytes = 0
    peak_global_used_bytes = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, int(training["epochs"]) + 1):
        epoch_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        loss_sum = 0.0
        for step, batch in enumerate(train_loader, start=1):
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                loss = model(**inputs).loss / accumulation
            scaler.scale(loss).backward()
            loss_sum += float(loss.detach().cpu()) * accumulation
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), float(training["max_grad_norm"]))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        predicted, _, validation_loss = _evaluate(model, validation_loader, device)
        metrics = evaluate_binary(validation.y.tolist(), predicted)
        epoch_peak_allocated = int(torch.cuda.max_memory_allocated(device))
        epoch_peak_reserved = int(torch.cuda.max_memory_reserved(device))
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        global_used = int(total_memory - free_memory)
        peak_allocated_bytes = max(peak_allocated_bytes, epoch_peak_allocated)
        peak_reserved_bytes = max(peak_reserved_bytes, epoch_peak_reserved)
        peak_global_used_bytes = max(peak_global_used_bytes, global_used)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / len(train_loader),
            "validation_loss": validation_loss,
            **metrics,
            "runtime_seconds": time.perf_counter() - epoch_started,
            "learning_rate_after_epoch": scheduler.get_last_lr()[0],
            "gpu_peak_allocated_gib": epoch_peak_allocated / 2**30,
            "gpu_peak_reserved_gib": epoch_peak_reserved / 2**30,
            "gpu_global_used_gib_at_epoch_end": global_used / 2**30,
        }
        epochs.append(record)
        if is_better_checkpoint(metrics, validation_loss, epoch, best_key):
            best_key = checkpoint_selection_key(metrics, validation_loss, epoch)
            best_epoch = epoch
            no_improvement = 0
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint_dir)
        else:
            no_improvement += 1
            if no_improvement >= int(training["early_stopping_patience"]):
                break

    del optimizer, scheduler, scaler, model
    torch.cuda.empty_cache()
    best_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
    predictions, fake_probabilities, best_validation_loss = _evaluate(
        best_model, validation_loader, device
    )
    best_metrics = evaluate_binary(validation.y.tolist(), predictions)
    best_metrics["validation_loss"] = best_validation_loss
    prediction_frame = pd.DataFrame(
        {
            "id": validation.metadata["Id"],
            "y_true": validation.y,
            "y_pred": predictions,
            "probability_fake": fake_probabilities,
        }
    )
    prediction_frame["correct"] = prediction_frame["y_true"].eq(prediction_frame["y_pred"])
    prediction_frame.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8")
    pd.DataFrame(epochs).to_csv(run_dir / "history.csv", index=False, encoding="utf-8")
    write_json(
        run_dir / "confusion_matrix.json",
        {
            "label_order": ["True", "Fake"],
            "positive_class": "Fake",
            "positive_label": 1,
            "matrix": best_metrics["confusion_matrix"],
            "tn": best_metrics["tn"],
            "fp": best_metrics["fp"],
            "fn": best_metrics["fn"],
            "tp": best_metrics["tp"],
        },
    )
    write_json(
        run_dir / "metrics.json",
        {"split": "development_experimental", "best_epoch": best_epoch, "selection_metric": "f1", "metrics": best_metrics, "test_metrics": None},
    )
    write_json(
        run_dir / "selection.json",
        {"best_epoch": best_epoch, "criterion": "maximum development F1(Fake=1); exact tie: higher recall, lower validation loss, earlier epoch", "selected_before_test": True, "test_accessed": False, "test_accessed_for_model_selection": False, "dataset_fingerprint": experimental_fingerprint},
    )
    runtime = time.perf_counter() - started
    hardware["training_memory"] = {
        "pytorch_peak_allocated_gib": peak_allocated_bytes / 2**30,
        "pytorch_peak_reserved_gib": peak_reserved_bytes / 2**30,
        "maximum_global_gpu_used_gib_observed": peak_global_used_bytes / 2**30,
        "global_measurement_includes_other_processes": True,
    }
    write_json(run_dir / "hardware.json", hardware)
    write_json(
        run_dir / "run_summary.json",
        {"experiment_id": experiment_id, "timestamp_utc": timestamp.isoformat(), "model_family": "BETO", "model_checkpoint": model_config["checkpoint"], "revision": model_config["revision"], "config_hash": effective_config_hash, "status": status, "phase": "development", "runtime_seconds": runtime, "best_epoch": best_epoch, "metrics": best_metrics, "test_accessed": False, "test_accessed_for_model_selection": False, "dataset_fingerprint": experimental_fingerprint},
    )
    (run_dir / "run.log").write_text(
        "\n".join(
            [
                f"experiment_id={experiment_id}",
                f"checkpoint={model_config['checkpoint']}",
                f"revision={model_config['revision']}",
                f"config_hash={effective_config_hash}",
                f"status={status}",
                f"best_epoch={best_epoch}",
                f"selection_f1={best_metrics['f1']:.12f}",
                f"runtime_seconds={runtime:.6f}",
                "test_accessed=false",
                "test_accessed_for_model_selection=false",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    from .visualization import generate_transformer_figures

    figures = generate_transformer_figures(run_dir)
    if registry_path is not None:
        append_experiment_registry(
            registry_path,
            {
                "experiment_id": experiment_id,
                "timestamp": timestamp.isoformat(),
                "model_family": "BETO",
                "model_checkpoint": f"{model_config['checkpoint']}@{model_config['revision']}",
                "dataset_fingerprint": experimental_fingerprint,
                "config_hash": effective_config_hash,
                "seed": seed,
                "phase": "development",
                "status": status,
                "accuracy": best_metrics["accuracy"],
                "precision": best_metrics["precision"],
                "recall": best_metrics["recall"],
                "f1": best_metrics["f1"],
                "best_epoch": best_epoch,
                "runtime_seconds": runtime,
                "test_accessed_for_model_selection": False,
                "accessed_for_final_evaluation": False,
                "notes": "single fixed BETO configuration; TEST quarantined",
            },
        )
        from .visualization import plot_model_comparison

        figures.append(
            plot_model_comparison(registry_path, run_dir / "figures" / "model_comparison.png")
        )
    write_json(
        run_dir / "figures.json",
        {"generated": [path.relative_to(run_dir).as_posix() for path in figures]},
    )
    return run_dir
