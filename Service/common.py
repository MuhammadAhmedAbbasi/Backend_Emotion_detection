"""Shared helpers for realtime emotion detection.

This file contains the small pieces that the Service layer needs from the
training pipeline, so Service code does not import directly from Algorithm.
"""

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from scipy import signal, stats


# ---------------------------------------------------------------------------
# Model and preprocessing settings
# ---------------------------------------------------------------------------

SERVICE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SERVICE_DIR.parent

TRAINED_MODEL_FILE = BACKEND_DIR / "Algorithm" / "saved_model" / "trained_model.pkl"

SAMPLING_RATE = 200
WINDOW_SECONDS = 4.0

FILTER_LOW_HZ = 1.0
FILTER_HIGH_HZ = 45.0
FILTER_ORDER = 4

SELECTED_SEED_CHANNELS = ["TP7", "TP8", "AF3", "AF4"]

LABEL_NAMES = {
    0: "neutral",
    1: "sad",
    2: "fear",
    3: "happy",
}

EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

EPS = 1e-12


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def integrate_power(values: np.ndarray, frequencies: np.ndarray) -> float:
    """Integrate power spectral density values over a frequency range."""
    if values.size < 2:
        return 0.0
    return float(np.trapezoid(values, frequencies))


def compute_band_powers(
    channel_signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    bands: dict[str, tuple[float, float]] = EEG_BANDS,
) -> dict[str, float]:
    """Calculate Welch power for each EEG frequency band."""
    segment_length = min(len(channel_signal), 2 * sampling_rate)

    frequencies, psd = signal.welch(
        channel_signal,
        fs=sampling_rate,
        nperseg=segment_length,
    )

    band_powers = {}

    for band_name, (low, high) in bands.items():
        mask = (frequencies >= low) & (frequencies < high)

        band_powers[band_name] = (
            integrate_power(psd[mask], frequencies[mask])
            if mask.any()
            else 0.0
        )

    return band_powers


def hjorth_parameters(
    channel_signal: np.ndarray,
) -> tuple[float, float, float]:
    """Return Hjorth activity, mobility, and complexity."""
    x = np.asarray(channel_signal, dtype=np.float64)

    first_derivative = np.diff(x)
    second_derivative = np.diff(first_derivative)

    activity = float(np.var(x))
    first_variance = float(np.var(first_derivative))
    second_variance = float(np.var(second_derivative))

    mobility = np.sqrt(first_variance / (activity + EPS))
    derivative_mobility = np.sqrt(
        second_variance / (first_variance + EPS)
    )
    complexity = derivative_mobility / (mobility + EPS)

    return activity, float(mobility), float(complexity)


def differential_entropy(variance: float) -> float:
    """Calculate Gaussian differential entropy from a variance value."""
    safe_variance = max(variance, EPS)
    return float(
        0.5 * np.log(2.0 * np.pi * np.e * safe_variance)
    )


def statistical_features(
    channel_signal: np.ndarray,
) -> dict[str, float]:
    """Calculate basic statistical features for one EEG channel."""
    x = np.asarray(channel_signal, dtype=np.float64)
    centered = x - np.mean(x)

    zero_crossings = np.count_nonzero(
        np.diff(np.signbit(centered))
    )

    return {
        "mean": float(np.mean(x)),
        "variance": float(np.var(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ptp": float(np.ptp(x)),
        "skewness": float(
            stats.skew(x, bias=False, nan_policy="omit")
        ),
        "kurtosis": float(
            stats.kurtosis(x, bias=False, nan_policy="omit")
        ),
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "zero_crossing_rate": float(
            zero_crossings / max(1, len(x) - 1)
        ),
    }


def extract_channel_features(
    channel_signal: np.ndarray,
    channel_name: str,
    sampling_rate: int = SAMPLING_RATE,
) -> tuple[list[float], list[str]]:
    """Extract all model features for one EEG channel."""
    values = []
    names = []

    for feature_name, value in statistical_features(
        channel_signal
    ).items():
        names.append(f"{channel_name}_{feature_name}")
        values.append(value)

    activity, mobility, complexity = hjorth_parameters(
        channel_signal
    )

    extra_features = {
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "differential_entropy": differential_entropy(activity),
    }

    for feature_name, value in extra_features.items():
        names.append(f"{channel_name}_{feature_name}")
        values.append(value)

    band_powers = compute_band_powers(
        channel_signal,
        sampling_rate=sampling_rate,
    )

    total_power = sum(band_powers.values()) + EPS

    names.append(f"{channel_name}_total_band_power")
    values.append(float(total_power))

    for band_name, power in band_powers.items():
        band_features = {
            "power": float(power),
            "relative_power": float(power / total_power),
            "log_power": float(np.log(power + EPS)),
            "differential_entropy": differential_entropy(power),
        }

        for feature_name, value in band_features.items():
            names.append(
                f"{channel_name}_{band_name}_{feature_name}"
            )
            values.append(value)

    return values, names


def extract_feature_vector(
    window: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    channel_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Convert one channels x samples EEG window into a feature vector."""
    eeg_window = np.asarray(window, dtype=np.float64)

    if eeg_window.ndim != 2:
        raise ValueError(
            f"Expected a 2D channels x samples window, got {eeg_window.shape}"
        )

    if channel_names is None:
        channel_names = SELECTED_SEED_CHANNELS

    if len(channel_names) != eeg_window.shape[0]:
        raise ValueError(
            f"Expected {len(channel_names)} channels, "
            f"but received window shape {eeg_window.shape}"
        )

    feature_values = []
    feature_names = []

    for channel_index, channel_name in enumerate(channel_names):
        values, names = extract_channel_features(
            eeg_window[channel_index],
            channel_name,
            sampling_rate,
        )

        feature_values.extend(values)
        feature_names.extend(names)

    return (
        np.asarray(feature_values, dtype=np.float32),
        feature_names,
    )
