"""Feature extraction for SEED-IV/Muse-compatible EEG windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import signal, stats

from Algorithm import config

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is a convenience only.
    tqdm = None


EPS = 1e-12


def _simpson_or_trapezoid(values: np.ndarray, freqs: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.trapezoid(values, freqs))


def compute_band_powers(
    channel_signal: np.ndarray,
    sampling_rate: int = config.SAMPLING_RATE,
    bands: dict[str, tuple[float, float]] = config.EEG_BANDS,
) -> dict[str, float]:
    """Compute absolute Welch band powers for one EEG channel."""
    nperseg = min(len(channel_signal), int(2 * sampling_rate))
    freqs, psd = signal.welch(channel_signal, fs=sampling_rate, nperseg=nperseg)
    powers: dict[str, float] = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        powers[band_name] = _simpson_or_trapezoid(psd[mask], freqs[mask]) if mask.any() else 0.0
    return powers


def hjorth_parameters(channel_signal: np.ndarray) -> tuple[float, float, float]:
    """Return Hjorth activity, mobility, and complexity."""
    x = np.asarray(channel_signal, dtype=np.float64)
    dx = np.diff(x)
    ddx = np.diff(dx)

    activity = float(np.var(x))
    var_dx = float(np.var(dx))
    var_ddx = float(np.var(ddx))

    mobility = np.sqrt(var_dx / (activity + EPS))
    mobility_dx = np.sqrt(var_ddx / (var_dx + EPS))
    complexity = mobility_dx / (mobility + EPS)
    return activity, float(mobility), float(complexity)


def differential_entropy_from_variance(variance: float) -> float:
    """Gaussian differential entropy using signal variance."""
    return float(0.5 * np.log(2.0 * np.pi * np.e * max(variance, EPS)))


def statistical_features(channel_signal: np.ndarray) -> dict[str, float]:
    """Return lightweight statistical features for one EEG channel."""
    x = np.asarray(channel_signal, dtype=np.float64)
    centered = x - np.mean(x)
    zero_crossings = np.count_nonzero(np.diff(np.signbit(centered)))
    return {
        "mean": float(np.mean(x)),
        "variance": float(np.var(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "ptp": float(np.ptp(x)),
        "skewness": float(stats.skew(x, bias=False, nan_policy="omit")),
        "kurtosis": float(stats.kurtosis(x, bias=False, nan_policy="omit")),
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "zero_crossing_rate": float(zero_crossings / max(1, len(x) - 1)),
    }


def extract_channel_features(
    channel_signal: np.ndarray,
    channel_name: str,
    sampling_rate: int = config.SAMPLING_RATE,
) -> tuple[list[float], list[str]]:
    """Extract all features for one channel."""
    values: list[float] = []
    names: list[str] = []

    stats_features = statistical_features(channel_signal)
    for feature_name, value in stats_features.items():
        names.append(f"{channel_name}_{feature_name}")
        values.append(value)

    activity, mobility, complexity = hjorth_parameters(channel_signal)
    for feature_name, value in {
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "differential_entropy": differential_entropy_from_variance(activity),
    }.items():
        names.append(f"{channel_name}_{feature_name}")
        values.append(value)

    band_powers = compute_band_powers(channel_signal, sampling_rate=sampling_rate)
    total_power = sum(band_powers.values()) + EPS
    names.append(f"{channel_name}_total_band_power")
    values.append(float(total_power))

    for band_name, power in band_powers.items():
        names.append(f"{channel_name}_{band_name}_power")
        values.append(float(power))
        names.append(f"{channel_name}_{band_name}_relative_power")
        values.append(float(power / total_power))
        names.append(f"{channel_name}_{band_name}_log_power")
        values.append(float(np.log(power + EPS)))
        names.append(f"{channel_name}_{band_name}_differential_entropy")
        values.append(differential_entropy_from_variance(power))

    return values, names


def extract_feature_vector(
    window: np.ndarray,
    sampling_rate: int = config.SAMPLING_RATE,
    channel_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract a 1D feature vector from one channels x samples EEG window.

    Use this exact function for both training and live Muse inference so the
    feature layout remains identical.
    """
    eeg_window = np.asarray(window, dtype=np.float64)
    if eeg_window.ndim != 2:
        raise ValueError(f"Expected a 2D channels x samples window, got shape {eeg_window.shape}")

    if channel_names is None:
        channel_names = config.SELECTED_SEED_CHANNELS
    if len(channel_names) != eeg_window.shape[0]:
        raise ValueError(
            f"Expected {len(channel_names)} channels from channel_names, got window shape {eeg_window.shape}"
        )

    all_values: list[float] = []
    all_names: list[str] = []
    for channel_idx, channel_name in enumerate(channel_names):
        values, names = extract_channel_features(
            eeg_window[channel_idx],
            channel_name=channel_name,
            sampling_rate=sampling_rate,
        )
        all_values.extend(values)
        all_names.extend(names)

    return np.asarray(all_values, dtype=np.float32), all_names


def extract_feature_matrix(
    windows: np.ndarray,
    sampling_rate: int = config.SAMPLING_RATE,
    channel_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract feature matrix from processed windows."""
    if windows.ndim != 3:
        raise ValueError(f"Expected windows with shape n_windows x channels x samples, got {windows.shape}")

    iterator = range(windows.shape[0])
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Extracting EEG features")

    rows: list[np.ndarray] = []
    feature_names: list[str] | None = None
    for idx in iterator:
        row, names = extract_feature_vector(
            windows[idx],
            sampling_rate=sampling_rate,
            channel_names=channel_names,
        )
        rows.append(row)
        if feature_names is None:
            feature_names = names

    return np.vstack(rows).astype(np.float32), feature_names or []


def save_features(
    processed_file: Path = config.PROCESSED_WINDOWS_FILE,
    output_file: Path = config.FEATURES_FILE,
) -> Path:
    """Load processed windows, extract features, and save the matrix."""
    if not processed_file.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_file}. Run dataset_preparation.py first."
        )

    data = np.load(processed_file, allow_pickle=True)
    windows = data["X"]
    labels = data["y"]
    groups = data["groups"] if "groups" in data.files else np.arange(len(labels))
    sampling_rate = int(data["sampling_rate"]) if "sampling_rate" in data.files else config.SAMPLING_RATE
    selected_seed_channels = (
        [str(ch) for ch in data["selected_seed_channels"]]
        if "selected_seed_channels" in data.files
        else config.SELECTED_SEED_CHANNELS
    )

    X_features, feature_names = extract_feature_matrix(
        windows,
        sampling_rate=sampling_rate,
        channel_names=selected_seed_channels,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "sampling_rate_hz": sampling_rate,
        "selected_seed_channels": selected_seed_channels,
        "selected_muse_channels": config.SELECTED_MUSE_CHANNELS,
        "feature_count": len(feature_names),
        "label_names": config.LABEL_NAMES,
        "source_processed_file": str(processed_file),
    }
    np.savez_compressed(
        output_file,
        X=X_features,
        y=labels,
        groups=groups,
        feature_names=np.array(feature_names),
        metadata_json=np.array(json.dumps(metadata)),
    )
    print(f"Saved features: {output_file}")
    print(f"Feature matrix shape: X={X_features.shape}, y={labels.shape}")
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract EEG features from processed SEED-IV windows.")
    parser.add_argument("--processed", type=Path, default=config.PROCESSED_WINDOWS_FILE)
    parser.add_argument("--output", type=Path, default=config.FEATURES_FILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_features(processed_file=args.processed, output_file=args.output)


if __name__ == "__main__":
    main()
