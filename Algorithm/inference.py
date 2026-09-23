"""Inference helpers for trained SEED-IV/Muse-compatible emotion models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from Algorithm import config
from Algorithm.feature_extraction import extract_feature_vector


def load_model(model_file: Path = config.TRAINED_MODEL_FILE) -> dict[str, Any]:
    if not model_file.exists():
        raise FileNotFoundError(f"Trained model not found at {model_file}. Run train_model.py first.")
    artifact = joblib.load(model_file)
    if "model" not in artifact:
        raise ValueError("Invalid model artifact: missing 'model'.")
    return artifact


def _probability_map(model: Any, probabilities: np.ndarray | None, label_names: dict[int, str]) -> dict[str, float] | None:
    if probabilities is None:
        return None

    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        last_step = list(model.named_steps.values())[-1]
        classes = getattr(last_step, "classes_", None)
    if classes is None:
        classes = np.arange(probabilities.shape[0])

    return {
        label_names.get(int(class_id), str(class_id)): float(probabilities[idx])
        for idx, class_id in enumerate(classes)
    }


def predict_from_feature_vector(
    feature_vector: np.ndarray,
    model_file: Path = config.TRAINED_MODEL_FILE,
) -> dict[str, Any]:
    """Predict emotion from a processed 1D feature vector."""
    artifact = load_model(model_file)
    model = artifact["model"]
    label_names = {int(key): value for key, value in artifact.get("label_names", config.LABEL_NAMES).items()}

    features = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
    expected_features = len(artifact.get("feature_names", []))
    if expected_features and features.shape[1] != expected_features:
        raise ValueError(f"Expected {expected_features} features, got {features.shape[1]}.")

    class_id = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None
    return {
        "predicted_class": class_id,
        "predicted_label": label_names.get(class_id, str(class_id)),
        "probabilities": _probability_map(model, probabilities, label_names),
        "model_name": artifact.get("model_name"),
    }


def predict_from_eeg_window(
    window: np.ndarray,
    model_file: Path = config.TRAINED_MODEL_FILE,
) -> dict[str, Any]:
    """Extract features from a channels x samples EEG window, then predict."""
    artifact = load_model(model_file)
    channel_names = artifact.get("selected_seed_channels", config.SELECTED_SEED_CHANNELS)
    sampling_rate = int(artifact.get("sampling_rate_hz", config.SAMPLING_RATE))
    features, _ = extract_feature_vector(
        window,
        sampling_rate=sampling_rate,
        channel_names=channel_names,
    )
    return predict_from_feature_vector(features, model_file=model_file)


def load_feature_vector(path: Path) -> np.ndarray:
    """Load a feature vector from .npy, .json, .csv, or .txt."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("features")
        return np.asarray(payload, dtype=np.float32)
    if suffix in {".csv", ".txt"}:
        return np.genfromtxt(path, delimiter=",", dtype=np.float32)
    raise ValueError(f"Unsupported feature vector file type: {path.suffix}")


def parse_feature_string(feature_string: str) -> np.ndarray:
    return np.asarray([float(value.strip()) for value in feature_string.split(",") if value.strip()], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run emotion inference with a trained EEG model.")
    parser.add_argument("--model", type=Path, default=config.TRAINED_MODEL_FILE)
    parser.add_argument("--features", type=str, default=None, help="Comma-separated processed feature vector.")
    parser.add_argument("--features-file", type=Path, default=None, help="Path to .npy/.json/.csv feature vector.")
    parser.add_argument("--window-npy", type=Path, default=None, help="Optional channels x samples EEG window .npy file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provided = [args.features is not None, args.features_file is not None, args.window_npy is not None]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of --features, --features-file, or --window-npy.")

    if args.features is not None:
        result = predict_from_feature_vector(parse_feature_string(args.features), model_file=args.model)
    elif args.features_file is not None:
        result = predict_from_feature_vector(load_feature_vector(args.features_file), model_file=args.model)
    else:
        result = predict_from_eeg_window(np.load(args.window_npy), model_file=args.model)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
