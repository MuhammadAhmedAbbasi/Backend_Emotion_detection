"""Run emotion prediction with a trained EEG model."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from Algorithm.feature_extraction import extract_feature_vector


# Default model settings
ALGORITHM_DIR = Path(__file__).resolve().parent
TRAINED_MODEL_FILE = ALGORITHM_DIR / "saved_model" / "trained_model.pkl"

SAMPLING_RATE = 200
SELECTED_SEED_CHANNELS = ["TP7", "TP8", "AF3", "AF4"]

LABEL_NAMES = {
    0: "neutral",
    1: "sad",
    2: "fear",
    3: "happy",
}


def load_model(model_file: Path = TRAINED_MODEL_FILE) -> dict[str, Any]:
    """Load and validate the saved model artifact."""
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_file}. Run train_model.py first."
        )

    artifact = joblib.load(model_file)

    if "model" not in artifact:
        raise ValueError("Invalid model artifact: missing 'model'.")

    return artifact


def probability_map(
    model: Any,
    probabilities: np.ndarray | None,
    label_names: dict[int, str],
) -> dict[str, float] | None:
    """Convert model probability output into emotion-name probabilities."""
    if probabilities is None:
        return None

    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        final_step = list(model.named_steps.values())[-1]
        classes = getattr(final_step, "classes_", None)

    if classes is None:
        classes = np.arange(len(probabilities))

    return {
        label_names.get(int(class_id), str(class_id)): float(probabilities[index])
        for index, class_id in enumerate(classes)
    }


def predict_from_feature_vector(
    feature_vector: np.ndarray,
    model_file: Path = TRAINED_MODEL_FILE,
) -> dict[str, Any]:
    """Predict an emotion from one processed feature vector."""
    artifact = load_model(model_file)
    model = artifact["model"]

    label_names = {
        int(key): value
        for key, value in artifact.get("label_names", LABEL_NAMES).items()
    }

    features = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)

    expected_count = len(artifact.get("feature_names", []))
    if expected_count and features.shape[1] != expected_count:
        raise ValueError(
            f"Expected {expected_count} features, got {features.shape[1]}."
        )

    predicted_class = int(model.predict(features)[0])

    probabilities = (
        model.predict_proba(features)[0]
        if hasattr(model, "predict_proba")
        else None
    )

    return {
        "predicted_class": predicted_class,
        "predicted_label": label_names.get(
            predicted_class,
            str(predicted_class),
        ),
        "probabilities": probability_map(
            model,
            probabilities,
            label_names,
        ),
        "model_name": artifact.get("model_name"),
    }


def predict_from_eeg_window(
    window: np.ndarray,
    model_file: Path = TRAINED_MODEL_FILE,
) -> dict[str, Any]:
    """Extract features from one EEG window and predict its emotion."""
    artifact = load_model(model_file)

    channel_names = artifact.get(
        "selected_seed_channels",
        SELECTED_SEED_CHANNELS,
    )
    sampling_rate = int(
        artifact.get("sampling_rate_hz", SAMPLING_RATE)
    )

    features, _ = extract_feature_vector(
        window,
        sampling_rate=sampling_rate,
        channel_names=channel_names,
    )

    return predict_from_feature_vector(
        features,
        model_file=model_file,
    )


def load_feature_vector(path: Path) -> np.ndarray:
    """Load a feature vector from .npy, .json, .csv, or .txt."""
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return np.load(path)

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("features")
        return np.asarray(data, dtype=np.float32)

    if suffix in {".csv", ".txt"}:
        return np.genfromtxt(path, delimiter=",", dtype=np.float32)

    raise ValueError(
        f"Unsupported feature vector file type: {path.suffix}"
    )


def parse_feature_string(feature_string: str) -> np.ndarray:
    """Convert comma-separated feature values into a NumPy array."""
    values = [
        float(value.strip())
        for value in feature_string.split(",")
        if value.strip()
    ]
    return np.asarray(values, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run emotion inference with a trained EEG model."
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=TRAINED_MODEL_FILE,
    )
    parser.add_argument(
        "--features",
        type=str,
        help="Comma-separated processed feature vector.",
    )
    parser.add_argument(
        "--features-file",
        type=Path,
        help="Path to a .npy, .json, .csv, or .txt feature vector.",
    )
    parser.add_argument(
        "--window-npy",
        type=Path,
        help="Path to a channels x samples EEG window saved as .npy.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    inputs_provided = sum(
        value is not None
        for value in (
            args.features,
            args.features_file,
            args.window_npy,
        )
    )

    if inputs_provided != 1:
        raise ValueError(
            "Provide exactly one of --features, "
            "--features-file, or --window-npy."
        )

    if args.features is not None:
        feature_vector = parse_feature_string(args.features)
        result = predict_from_feature_vector(
            feature_vector,
            model_file=args.model,
        )

    elif args.features_file is not None:
        feature_vector = load_feature_vector(args.features_file)
        result = predict_from_feature_vector(
            feature_vector,
            model_file=args.model,
        )

    else:
        window = np.load(args.window_npy)
        result = predict_from_eeg_window(
            window,
            model_file=args.model,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
