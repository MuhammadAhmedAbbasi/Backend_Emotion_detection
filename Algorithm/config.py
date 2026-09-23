"""Simple settings shared by training and realtime detection."""

from pathlib import Path


ALGORITHM_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ALGORITHM_DIR.parent
WORKSPACE_DIR = BACKEND_DIR.parent

# Change only this line if the SEED-IV dataset is moved.
RAW_EEG_DIR = WORKSPACE_DIR / "archive" / "seed_iv" / "eeg_raw_data"

PROCESSED_DATA_DIR = ALGORITHM_DIR / "processed_data"
FEATURES_DIR = ALGORITHM_DIR / "features"
SAVED_MODEL_DIR = ALGORITHM_DIR / "saved_model"

PROCESSED_WINDOWS_FILE = PROCESSED_DATA_DIR / "processed_windows.npz"
PROCESSED_METADATA_FILE = PROCESSED_DATA_DIR / "metadata.json"
FEATURES_FILE = FEATURES_DIR / "features.npz"
TRAINED_MODEL_FILE = SAVED_MODEL_DIR / "trained_model.pkl"

# SEED-IV raw EEG uses 62 electrodes in this order.
SEED_IV_CHANNELS = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1",
    "OZ", "O2", "CB2",
]

# Muse has TP9, TP10, AF7, AF8. SEED-IV lacks those exact positions, so we use
# the closest electrodes available in the SEED-IV 62-channel montage.
MUSE_TO_SEED_CHANNELS = {
    "TP9": {
        "seed_channel": "TP7",
        "reason": "nearest left temporal-parietal SEED-IV electrode to the left mastoid/ear TP9 site",
    },
    "TP10": {
        "seed_channel": "TP8",
        "reason": "nearest right temporal-parietal SEED-IV electrode to the right mastoid/ear TP10 site",
    },
    "AF7": {
        "seed_channel": "AF3",
        "reason": "nearest left anterior-frontal SEED-IV electrode; F7 is more lateral/inferior",
    },
    "AF8": {
        "seed_channel": "AF4",
        "reason": "nearest right anterior-frontal SEED-IV electrode; F8 is more lateral/inferior",
    },
}

SELECTED_MUSE_CHANNELS = list(MUSE_TO_SEED_CHANNELS.keys())
SELECTED_SEED_CHANNELS = [
    MUSE_TO_SEED_CHANNELS[muse_channel]["seed_channel"]
    for muse_channel in SELECTED_MUSE_CHANNELS
]
SELECTED_CHANNEL_INDICES = [
    SEED_IV_CHANNELS.index(seed_channel) for seed_channel in SELECTED_SEED_CHANNELS
]

SAMPLING_RATE = 200
WINDOW_SECONDS = 4.0
WINDOW_OVERLAP = 0.5

APPLY_BANDPASS_FILTER = True
FILTER_LOW_HZ = 1.0
FILTER_HIGH_HZ = 45.0
FILTER_ORDER = 4

EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

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

TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_SELECTION_METRIC = "macro_f1"
