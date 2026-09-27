"""Research-gated PyTorch sequence regressors under the SupervisedModel contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from trading_pipeline.experiments import ComponentSpec, FitContext, SupervisedModel
from trading_pipeline.features.sequence_view import SequenceBatch


def _space(family: str) -> Mapping[str, Any]:
    parameters: dict[str, Any] = {
        "lookback": {"type": "categorical", "values": [10, 20, 40]},
        "depth": {"type": "categorical", "values": [1, 2]},
        "width": {"type": "categorical", "values": [32, 64]},
        "dropout": {"type": "categorical", "values": [0.0, 0.2]},
        "activation": {"type": "categorical", "values": ["relu", "gelu"]},
        "optimizer": {"type": "categorical", "values": ["adam", "adamw"]},
        "learning_rate": {"type": "loguniform", "low": 1e-4, "high": 3e-3},
        # The expanded universe makes 64/128-row CUDA batches transfer-bound.
        # These sizes remain small relative to 16 GiB VRAM while exercising a
        # meaningful optimisation range at the declared production scale.
        "batch_size": {"type": "categorical", "values": [512, 1024]},
    }
    if family == "transformer":
        parameters["heads"] = {"type": "categorical", "values": [2, 4]}
        parameters["feed_forward"] = {"type": "categorical", "values": [64, 128]}
    return {"schema_version": 1, "strategy": "random_screen", "resource_axis": "epochs", "parameters": parameters}


LSTM_SPEC = ComponentSpec(
    component_id="supervised.lstm.v1", interface="SupervisedModel",
    implementation="trading_pipeline.modelling.supervised_models.deep_sequence.LSTMModel",
    research_enabled=False,
    capabilities={"study_adapter": True, "train_only_preprocessing": True, "stopping_data": True,
                  "continuation": False, "staged_metrics": True,
                  "optional_dependency": True, "cpu": True, "cuda": True,
                  "sequence_view": True},
)
TRANSFORMER_SPEC = ComponentSpec(
    component_id="supervised.causal_transformer.v1", interface="SupervisedModel",
    implementation="trading_pipeline.modelling.supervised_models.deep_sequence.CausalTransformerModel",
    research_enabled=False,
    capabilities={"study_adapter": True, "train_only_preprocessing": True, "stopping_data": True,
                  "continuation": False, "staged_metrics": True,
                  "optional_dependency": True, "cpu": True, "cuda": True,
                  "sequence_view": True},
)


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for deep supervised models; install the optional torch dependency") from exc
    return torch, nn


def _network(family: str, features: int, params: Mapping[str, Any]):
    torch, nn = _torch()
    activation = {"relu": nn.ReLU, "gelu": nn.GELU}[params["activation"]]
    width, depth = params["width"], params["depth"]

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            # Explicit feature-observation channel distinguishes missing from zero.
            self.input = nn.Linear(features * 2, width)
            self.activation = activation()
            if family == "lstm":
                self.encoder = nn.LSTM(width, width, num_layers=depth, batch_first=True,
                                       dropout=params["dropout"] if depth > 1 else 0.0)
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=width, nhead=params["heads"],
                    dim_feedforward=params["feed_forward"], dropout=params["dropout"],
                    activation=params["activation"], batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
                self.position = nn.Embedding(params["lookback"], width)
            self.head = nn.Linear(width, 1)

        def forward(self, values, time_mask, feature_mask):
            length = time_mask.sum(dim=1)
            data = torch.cat((values * feature_mask.to(values.dtype),
                              feature_mask.to(values.dtype)), dim=-1)
            hidden = self.activation(self.input(data))
            if family == "lstm":
                packed = nn.utils.rnn.pack_padded_sequence(
                    hidden, length.cpu(), batch_first=True, enforce_sorted=False)
                _, (last, _) = self.encoder(packed)
                summary = last[-1]
            else:
                positions = torch.arange(values.shape[1], device=values.device)
                hidden = hidden + self.position(positions)[None, :, :]
                causal = torch.triu(torch.ones(values.shape[1], values.shape[1],
                                                device=values.device, dtype=torch.bool), diagonal=1)
                encoded = self.encoder(hidden, mask=causal, src_key_padding_mask=~time_mask)
                summary = encoded[torch.arange(len(length), device=values.device), length - 1]
            return self.head(summary).squeeze(-1)

    return Network()


class DeepSequenceModel(SupervisedModel):
    FAMILY: str
    SPEC: ComponentSpec

    def __init__(self, *, params: Mapping[str, Any] | None = None, device: str = "cpu",
                 training_config: Mapping[str, Any] | None = None) -> None:
        defaults = {"lookback": 20, "depth": 1, "width": 32, "dropout": 0.0,
                    "activation": "relu", "optimizer": "adamw", "learning_rate": 1e-3,
                    "batch_size": 64}
        if self.FAMILY == "transformer":
            defaults.update(heads=2, feed_forward=64)
        self._params = {**defaults, **dict(params or {})}
        self._validate_params()
        if device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be cpu, cuda or auto")
        self._requested_device = device
        self._training_config = {"epochs": 20, "patience": 5, **dict(training_config or {})}
        if (not isinstance(self._training_config["epochs"], int) or self._training_config["epochs"] < 1
                or not isinstance(self._training_config["patience"], int) or self._training_config["patience"] < 1):
            raise ValueError("epochs and patience must be positive integers")
        self._model = None
        self._feature_count: int | None = None
        self._actual_device: str | None = None
        self._telemetry: dict[str, Any] = {"status": "not_fitted", "parameters": dict(self._params)}

    @property
    def spec(self) -> ComponentSpec:
        return self.SPEC

    def search_space(self) -> Mapping[str, Any]:
        return deepcopy(_space(self.FAMILY))

    def _validate_params(self) -> None:
        p = self._params
        allowed = set(_space(self.FAMILY)["parameters"])
        if set(p) != allowed:
            raise ValueError(f"Deep model parameters must be exactly {sorted(allowed)}")
        for name in ("lookback", "depth", "width", "batch_size"):
            if not isinstance(p[name], int) or p[name] < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= p["dropout"] < 1 or not 0 < p["learning_rate"] <= 1:
            raise ValueError("Invalid dropout or learning rate")
        if p["activation"] not in {"relu", "gelu"} or p["optimizer"] not in {"adam", "adamw"}:
            raise ValueError("Unsupported activation or optimiser")
        if self.FAMILY == "transformer":
            if (not isinstance(p["heads"], int) or p["heads"] < 1
                    or p["width"] % p["heads"] != 0):
                raise ValueError("Transformer width must be divisible by positive heads")
            if not isinstance(p["feed_forward"], int) or p["feed_forward"] < 1:
                raise ValueError("feed_forward must be a positive integer")

    def _validate_batch(self, x, y=None) -> tuple[SequenceBatch, np.ndarray | None]:
        if not isinstance(x, SequenceBatch):
            raise ValueError("Deep models require a SequenceBatch from the causal feature view")
        if x.values.shape[1] != self._params["lookback"]:
            raise ValueError("Sequence lookback differs from model parameters")
        if self._feature_count is not None and x.values.shape[2] != self._feature_count:
            raise ValueError("Sequence feature count differs from fitted model")
        if y is None:
            return x, None
        target = np.asarray(y, dtype=np.float32)
        if target.ndim != 1 or len(target) != len(x.values) or not np.isfinite(target).all():
            raise ValueError("Targets must be finite, one-dimensional and row-aligned")
        return x, target

    def _tensors(self, batch: SequenceBatch, target, device, torch, indices):
        """Transfer only one bounded host batch to the selected device."""
        indices = np.asarray(indices, dtype=np.intp)
        if indices.ndim != 1 or not 0 < len(indices) <= self._params["batch_size"]:
            raise ValueError("Device transfer exceeds declared batch_size")
        tensors = [torch.as_tensor(batch.values[indices], device=device),
                   torch.as_tensor(batch.time_mask[indices], device=device),
                   torch.as_tensor(batch.feature_mask[indices], device=device)]
        if target is not None:
            tensors.append(torch.as_tensor(target[indices], device=device))
        return tensors

    def _mean_squared_error(self, model, batch: SequenceBatch, target, device, torch) -> float:
        """Aggregate sample-weighted MSE without a full-batch device allocation."""
        squared = torch.zeros((), device=device, dtype=torch.float64)
        for start in range(0, len(batch.values), self._params["batch_size"]):
            indices = np.arange(start, min(start + self._params["batch_size"],
                                           len(batch.values)))
            values, time_mask, feature_mask, actual = self._tensors(
                batch, target, device, torch, indices)
            prediction = model(values, time_mask, feature_mask)
            squared += torch.sum((prediction - actual) ** 2, dtype=torch.float64)
        return float((squared / len(batch.values)).item())

    def fit(self, x, y, *, context: FitContext, stopping_data=None):
        train, train_y = self._validate_batch(x, y)
        if stopping_data is None:
            raise ValueError("Deep models require distinct chronological stopping_data")
        try:
            stop_x, stop_y = stopping_data
        except (TypeError, ValueError) as exc:
            raise ValueError("stopping_data must be an (x, y) pair") from exc
        stop, stop_y = self._validate_batch(stop_x, stop_y)
        if len(train.values) == 0 or len(stop.values) == 0:
            raise ValueError("Fit and stopping batches must be nonempty")
        if train.values.shape[2] != stop.values.shape[2]:
            raise ValueError("Fit and stopping feature counts must agree")
        torch, _ = _torch()
        torch.manual_seed(context.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(context.seed)
        torch.use_deterministic_algorithms(True)
        fallback = None
        actual = "cpu"
        if self._requested_device in {"cuda", "auto"}:
            if torch.cuda.is_available():
                actual = "cuda"
            else:
                fallback = "CUDA unavailable; used CPU"
        self._feature_count = train.values.shape[2]
        model = _network(self.FAMILY, self._feature_count, self._params).to(actual)
        optimizer_type = torch.optim.AdamW if self._params["optimizer"] == "adamw" else torch.optim.Adam
        optimizer = optimizer_type(model.parameters(), lr=self._params["learning_rate"])
        generator = torch.Generator(device="cpu").manual_seed(context.seed)
        epochs = int(context.fidelity.get("epochs", self._training_config["epochs"]))
        if epochs < 1:
            raise ValueError("fidelity epochs must be positive")
        trace: list[dict[str, float | int]] = []
        best_loss = float("inf")
        best_state = None
        best_epoch = 0
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()
            order = torch.randperm(len(train.values), generator=generator)
            for indices in order.split(self._params["batch_size"]):
                values, time_mask, feature_mask, target = self._tensors(
                    train, train_y, actual, torch, indices.numpy())
                optimizer.zero_grad(set_to_none=True)
                pred = model(values, time_mask, feature_mask)
                loss = torch.mean((pred - target) ** 2)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                train_loss = self._mean_squared_error(model, train, train_y, actual, torch)
                validation_loss = self._mean_squared_error(model, stop, stop_y, actual, torch)
            if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
                raise ValueError("Deep training produced a nonfinite loss")
            trace.append({"epoch": epoch, "train_mse": float(train_loss),
                          "stopping_mse": float(validation_loss)})
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                best_state = deepcopy(model.state_dict())
            if epoch - best_epoch >= self._training_config["patience"]:
                break
        model.load_state_dict(best_state)
        self._model = model
        self._actual_device = actual
        self._telemetry = {
            "status": "fitted", "study_id": context.study_id, "trial_id": context.trial_id,
            "fold_id": context.fold_id, "seed": context.seed, "fidelity": dict(context.fidelity),
            "parameters": dict(self._params), "fit_rows": len(train.values),
            "stopping_rows": len(stop.values), "fit_columns": self._feature_count,
            "requested_device": self._requested_device, "actual_device": actual,
            "fallback_reason": fallback, "deterministic_algorithms": True,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "epochs_ran": len(trace), "selected_epoch": best_epoch,
            "best_stopping_mse": float(best_loss), "training_trace": trace,
            "fit_duration_seconds": time.perf_counter() - started,
        }
        return self

    def predict(self, x):
        if self._model is None:
            raise RuntimeError("Deep model must be fitted before prediction")
        batch, _ = self._validate_batch(x)
        torch, _ = _torch()
        self._model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(batch.values), self._params["batch_size"]):
                indices = np.arange(start, min(start + self._params["batch_size"],
                                               len(batch.values)))
                tensors = self._tensors(batch, None, self._actual_device, torch, indices)
                predictions.append(self._model(*tensors).cpu().numpy())
        return np.concatenate(predictions).astype(np.float64)

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("Deep model must be fitted before saving")
        torch, _ = _torch()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"family": self.FAMILY, "params": dict(self._params),
                    "training_config": dict(self._training_config),
                    "feature_count": self._feature_count,
                    "state_dict": {k: v.cpu() for k, v in self._model.state_dict().items()},
                    "telemetry": self._telemetry}, destination)

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu"):
        torch, _ = _torch()
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["family"] != cls.FAMILY:
            raise ValueError("Saved checkpoint belongs to a different model family")
        model = cls(params=state["params"], device=device,
                    training_config=state["training_config"])
        actual = "cuda" if device in {"cuda", "auto"} and torch.cuda.is_available() else "cpu"
        model._feature_count = int(state["feature_count"])
        model._model = _network(cls.FAMILY, model._feature_count, model._params).to(actual)
        model._model.load_state_dict(state["state_dict"])
        model._model.eval()
        model._actual_device = actual
        model._telemetry = dict(state["telemetry"])
        return model

    def telemetry(self) -> Mapping[str, Any]:
        return deepcopy(self._telemetry)


class LSTMModel(DeepSequenceModel):
    FAMILY = "lstm"
    SPEC = LSTM_SPEC


class CausalTransformerModel(DeepSequenceModel):
    FAMILY = "transformer"
    SPEC = TRANSFORMER_SPEC


COMPONENT_REGISTRATIONS = (
    (LSTM_SPEC, LSTMModel),
    (TRANSFORMER_SPEC, CausalTransformerModel),
)
