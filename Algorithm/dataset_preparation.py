"""Simple SEED-IV EEG data preparation.

This version is intentionally plain and easy to read:
- use a fixed dataset folder,
- load each `.mat` file directly,
- keep only the four Muse-matched channels,
- filter and window the signal,
- save a compact NumPy file for feature extraction.
"""

import json
import re

import numpy as np
import scipy.io as sio
from scipy import signal

from Algorithm import config


RAW_EEG_DIR = config.RAW_EEG_DIR
OUTPUT_FILE = config.PROCESSED_WINDOWS_FILE
METADATA_FILE = config.PROCESSED_METADATA_FILE

WINDOW_SECONDS = config.WINDOW_SECONDS
OVERLAP = config.WINDOW_OVERLAP
SAMPLING_RATE = config.SAMPLING_RATE
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLING_RATE)
STEP_SAMPLES = int(WINDOW_SAMPLES * (1.0 - OVERLAP))

SEED_CHANNELS = config.SEED_IV_CHANNELS
KEEP_CHANNELS = config.SELECTED_SEED_CHANNELS
KEEP_INDICES = [SEED_CHANNELS.index(name) for name in KEEP_CHANNELS]

_EEG_KEY_PATTERN = re.compile(r"^[^_]+_eeg(?P<trial>\d+)$", re.IGNORECASE)


def _find_trial_key(data: dict, trial_number: int, mat_path) -> str:
    """Find a subject-specific SEED-IV key such as ``tyc_eeg1``."""
    if trial_number < 1:
        raise ValueError(f"trial_number must be positive, got {trial_number}")

    matches = {
        int(match.group("trial")): key
        for key in data
        if (match := _EEG_KEY_PATTERN.match(key)) is not None
    }
    if trial_number not in matches:
        available = sorted(matches)
        raise KeyError(
            f"No EEG trial {trial_number} found in {mat_path}. "
            f"Available trial numbers: {available}"
        )
    return matches[trial_number]


def load_trial(mat_path, trial_number: int) -> np.ndarray:
    data = sio.loadmat(mat_path)
    key = _find_trial_key(data, trial_number, mat_path)
    trial = np.asarray(data[key], dtype=np.float64)
    if trial.shape[0] != len(SEED_CHANNELS) and trial.shape[1] == len(SEED_CHANNELS):
        trial = trial.T
    if trial.ndim != 2 or trial.shape[0] != len(SEED_CHANNELS):
        raise ValueError(
            f"Unexpected shape for {key} in {mat_path}: {trial.shape}; "
            f"expected ({len(SEED_CHANNELS)}, samples)"
        )
    return trial


def select_channels(trial: np.ndarray) -> np.ndarray:
    return trial[KEEP_INDICES, :]


def replace_bad_values(trial: np.ndarray) -> np.ndarray:
    cleaned = trial.copy()
    for channel_index in range(cleaned.shape[0]):
        channel = cleaned[channel_index]
        good = np.isfinite(channel)
        if good.any():
            fill_value = np.median(channel[good])
        else:
            fill_value = 0.0
        channel[~good] = fill_value
    return cleaned


def bandpass_filter(trial: np.ndarray) -> np.ndarray:
    low = 1.0 / (SAMPLING_RATE / 2.0)
    high = 45.0 / (SAMPLING_RATE / 2.0)
    sos = signal.butter(4, [low, high], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, trial, axis=1)


def make_windows(trial: np.ndarray, label: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_samples = trial.shape[1]
    if total_samples < WINDOW_SAMPLES:
        empty_windows = np.empty((0, trial.shape[0], WINDOW_SAMPLES), dtype=np.float32)
        empty_labels = np.empty((0,), dtype=np.int64)
        empty_starts = np.empty((0,), dtype=np.int64)
        return empty_windows, empty_labels, empty_starts

    starts = np.arange(0, total_samples - WINDOW_SAMPLES + 1, STEP_SAMPLES, dtype=np.int64)
    windows = []
    for start in starts:
        windows.append(trial[:, start : start + WINDOW_SAMPLES])

    X = np.asarray(windows, dtype=np.float32)
    y = np.full((len(windows),), label, dtype=np.int64)
    return X, y, starts


def process_trial(raw_trial: np.ndarray) -> np.ndarray:
    selected = select_channels(raw_trial)
    selected = replace_bad_values(selected)
    selected = signal.detrend(selected, axis=1, type="constant")
    try:
        selected = bandpass_filter(selected)
    except ValueError:
        # If filtering ever fails on a short or odd signal, keep the cleaned data.
        pass
    return selected.astype(np.float32)


def main() -> None:
    raw_root = RAW_EEG_DIR
    if not raw_root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {raw_root}\n"
            "Open Algorithm/config.py and correct RAW_EEG_DIR."
        )
    print(f"Using raw EEG folder: {raw_root}")
    print(f"Keeping only: {KEEP_CHANNELS}")

    all_windows = []
    all_labels = []
    all_groups = []
    metadata_rows = []
    group_id = 0

    for session_id in (1, 2, 3):
        session_dir = raw_root / str(session_id)
        labels = config.SESSION_LABELS[session_id]
        mat_files = sorted(session_dir.glob("*.mat"))

        for mat_path in mat_files:
            subject_id = mat_path.stem.split("_")[0]
            print(f"Session {session_id}, subject {subject_id}, file {mat_path.name}")

            for trial_number, label in enumerate(labels, start=1):
                raw_trial = load_trial(mat_path, trial_number)
                processed_trial = process_trial(raw_trial)
                windows, y, starts = make_windows(processed_trial, label)

                if len(windows) == 0:
                    continue

                groups = np.full((len(windows),), group_id, dtype=np.int64)
                all_windows.append(windows)
                all_labels.append(y)
                all_groups.append(groups)
                metadata_rows.append(
                    {
                        "session": session_id,
                        "subject": int(subject_id),
                        "trial": trial_number,
                        "label": int(label),
                        "label_name": config.LABEL_NAMES[int(label)],
                        "n_windows": int(len(windows)),
                        "first_start": int(starts[0]),
                        "last_start": int(starts[-1]),
                    }
                )
                group_id += 1

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    groups = np.concatenate(all_groups, axis=0)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y,
        groups=groups,
        selected_seed_channels=np.array(KEEP_CHANNELS),
        selected_muse_channels=np.array(config.SELECTED_MUSE_CHANNELS),
        sampling_rate=np.array(SAMPLING_RATE),
        window_seconds=np.array(WINDOW_SECONDS),
        overlap=np.array(OVERLAP),
    )

    METADATA_FILE.write_text(
        json.dumps(
            {
                "raw_root": str(raw_root),
                "sampling_rate_hz": SAMPLING_RATE,
                "window_seconds": WINDOW_SECONDS,
                "overlap": OVERLAP,
                "selected_seed_channels": KEEP_CHANNELS,
                "selected_muse_channels": config.SELECTED_MUSE_CHANNELS,
                "label_names": config.LABEL_NAMES,
                "rows": metadata_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved processed windows to {OUTPUT_FILE}")
    print(f"Saved metadata to {METADATA_FILE}")
    print(f"Final shape: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()
