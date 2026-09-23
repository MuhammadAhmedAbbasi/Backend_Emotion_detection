"""Simple SEED-IV EEG data preparation.

This script is intentionally self-contained and easy to read:
- define the required preprocessing settings here,
- load each SEED-IV .mat file directly,
- keep only the four Muse-matched channels,
- clean, filter, and window the signal,
- save the processed windows for feature extraction.
"""

import json
import re
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import signal


# Paths
ALGORITHM_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ALGORITHM_DIR.parent
WORKSPACE_DIR = BACKEND_DIR.parent

# Change only this line if the SEED-IV dataset is moved.
RAW_EEG_DIR = WORKSPACE_DIR / "archive" / "seed_iv" / "eeg_raw_data"

OUTPUT_FILE = ALGORITHM_DIR / "processed_data" / "processed_windows.npz"
METADATA_FILE = ALGORITHM_DIR / "processed_data" / "metadata.json"


# Preprocessing settings
SAMPLING_RATE = 200
WINDOW_SECONDS = 4.0
OVERLAP = 0.5

FILTER_LOW_HZ = 1.0
FILTER_HIGH_HZ = 45.0
FILTER_ORDER = 4

WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLING_RATE)
STEP_SAMPLES = int(WINDOW_SAMPLES * (1.0 - OVERLAP))


# SEED-IV raw EEG uses 62 electrodes in this order.
SEED_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1",
    "OZ", "O2", "CB2",
]

# Muse channels do not exactly exist in the SEED-IV montage.
# These are the closest SEED-IV channels used for training.
MUSE_TO_SEED_CHANNELS = {
    "TP9": "TP7",
    "TP10": "TP8",
    "AF7": "AF3",
    "AF8": "AF4",
}

SELECTED_MUSE_CHANNELS = list(MUSE_TO_SEED_CHANNELS.keys())
KEEP_CHANNELS = list(MUSE_TO_SEED_CHANNELS.values())
KEEP_INDICES = [SEED_CHANNELS.index(channel) for channel in KEEP_CHANNELS]


# SEED-IV emotion labels for the 24 trials in each session.
SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}

LABEL_NAMES = {
    0: "neutral",
    1: "sad",
    2: "fear",
    3: "happy",
}


_EEG_KEY_PATTERN = re.compile(r"^[^_]+_eeg(?P<trial>\d+)$", re.IGNORECASE)


def _find_trial_key(data: dict, trial_number: int, mat_path: Path) -> str:
    """Find a subject-specific SEED-IV key such as tyc_eeg1."""
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


def load_trial(mat_path: Path, trial_number: int) -> np.ndarray:
    """Load one EEG trial from a SEED-IV .mat file."""
    data = sio.loadmat(mat_path)
    key = _find_trial_key(data, trial_number, mat_path)
    trial = np.asarray(data[key], dtype=np.float64)

    # Keep the shape consistent: channels x samples.
    if trial.shape[0] != len(SEED_CHANNELS) and trial.shape[1] == len(SEED_CHANNELS):
        trial = trial.T

    if trial.ndim != 2 or trial.shape[0] != len(SEED_CHANNELS):
        raise ValueError(
            f"Unexpected shape for {key} in {mat_path}: {trial.shape}; "
            f"expected ({len(SEED_CHANNELS)}, samples)"
        )

    return trial


def select_channels(trial: np.ndarray) -> np.ndarray:
    """Keep only the four SEED-IV channels matched to Muse."""
    return trial[KEEP_INDICES, :]


def replace_bad_values(trial: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf values with the median of the same channel."""
    cleaned = trial.copy()

    for channel_index in range(cleaned.shape[0]):
        channel = cleaned[channel_index]
        good = np.isfinite(channel)
        fill_value = np.median(channel[good]) if good.any() else 0.0
        channel[~good] = fill_value

    return cleaned


def bandpass_filter(trial: np.ndarray) -> np.ndarray:
    """Apply a Butterworth bandpass filter."""
    nyquist = SAMPLING_RATE / 2.0
    low = FILTER_LOW_HZ / nyquist
    high = FILTER_HIGH_HZ / nyquist

    sos = signal.butter(
        FILTER_ORDER,
        [low, high],
        btype="bandpass",
        output="sos",
    )
    return signal.sosfiltfilt(sos, trial, axis=1)


def make_windows(
    trial: np.ndarray,
    label: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split one trial into overlapping fixed-length windows."""
    total_samples = trial.shape[1]

    if total_samples < WINDOW_SAMPLES:
        empty_windows = np.empty(
            (0, trial.shape[0], WINDOW_SAMPLES),
            dtype=np.float32,
        )
        empty_labels = np.empty((0,), dtype=np.int64)
        empty_starts = np.empty((0,), dtype=np.int64)
        return empty_windows, empty_labels, empty_starts

    starts = np.arange(
        0,
        total_samples - WINDOW_SAMPLES + 1,
        STEP_SAMPLES,
        dtype=np.int64,
    )

    windows = [
        trial[:, start : start + WINDOW_SAMPLES]
        for start in starts
    ]

    X = np.asarray(windows, dtype=np.float32)
    y = np.full(len(windows), label, dtype=np.int64)
    return X, y, starts


def process_trial(raw_trial: np.ndarray) -> np.ndarray:
    """Select, clean, detrend, and filter one raw EEG trial."""
    trial = select_channels(raw_trial)
    trial = replace_bad_values(trial)
    trial = signal.detrend(trial, axis=1, type="constant")

    try:
        trial = bandpass_filter(trial)
    except ValueError:
        # If filtering fails on an unusually short signal, keep the cleaned data.
        pass

    return trial.astype(np.float32)


def main() -> None:
    if not RAW_EEG_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {RAW_EEG_DIR}\n"
            "Update RAW_EEG_DIR near the top of Algorithm/dataset_preparation.py."
        )

    print(f"Using raw EEG folder: {RAW_EEG_DIR}")
    print(f"Keeping only: {KEEP_CHANNELS}")

    all_windows = []
    all_labels = []
    all_groups = []
    metadata_rows = []
    group_id = 0

    for session_id, labels in SESSION_LABELS.items():
        session_dir = RAW_EEG_DIR / str(session_id)
        mat_files = sorted(session_dir.glob("*.mat"))

        for mat_path in mat_files:
            subject_id = mat_path.stem.split("_")[0]
            print(
                f"Session {session_id}, subject {subject_id}, "
                f"file {mat_path.name}"
            )

            for trial_number, label in enumerate(labels, start=1):
                raw_trial = load_trial(mat_path, trial_number)
                processed_trial = process_trial(raw_trial)
                windows, y, starts = make_windows(processed_trial, label)

                if len(windows) == 0:
                    continue

                groups = np.full(len(windows), group_id, dtype=np.int64)

                all_windows.append(windows)
                all_labels.append(y)
                all_groups.append(groups)

                metadata_rows.append(
                    {
                        "session": session_id,
                        "subject": int(subject_id),
                        "trial": trial_number,
                        "label": int(label),
                        "label_name": LABEL_NAMES[int(label)],
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
        selected_muse_channels=np.array(SELECTED_MUSE_CHANNELS),
        sampling_rate=np.array(SAMPLING_RATE),
        window_seconds=np.array(WINDOW_SECONDS),
        overlap=np.array(OVERLAP),
    )

    metadata = {
        "raw_root": str(RAW_EEG_DIR),
        "sampling_rate_hz": SAMPLING_RATE,
        "window_seconds": WINDOW_SECONDS,
        "overlap": OVERLAP,
        "selected_seed_channels": KEEP_CHANNELS,
        "selected_muse_channels": SELECTED_MUSE_CHANNELS,
        "label_names": LABEL_NAMES,
        "rows": metadata_rows,
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved processed windows to {OUTPUT_FILE}")
    print(f"Saved metadata to {METADATA_FILE}")
    print(f"Final shape: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()
