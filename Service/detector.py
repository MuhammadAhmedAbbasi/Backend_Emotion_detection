"""Realtime emotion detector that bridges Muse windows to the trained model."""

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from Service.common import (
    FILTER_HIGH_HZ,
    FILTER_LOW_HZ,
    FILTER_ORDER,
    LABEL_NAMES,
    SAMPLING_RATE,
    SELECTED_SEED_CHANNELS,
    TRAINED_MODEL_FILE,
    WINDOW_SECONDS,
    extract_feature_vector,
    load_model,
)
from Service.muse_client import MUSE_SAMPLE_RATE, MuseReceiver


MUSE_CHANNEL_ORDER_FOR_MODEL = ["TP9", "TP10", "AF7", "AF8"]


class EmotionDetectionService:
    """Continuously turn recent Muse data into emotion predictions."""

    def __init__(
        self,
        receiver: MuseReceiver,
        model_file: Path = TRAINED_MODEL_FILE,
        step_seconds: float = 1.0,
    ) -> None:
        self.receiver = receiver
        self.model_file = model_file
        self.step_seconds = step_seconds

        self.window_seconds = WINDOW_SECONDS
        self.model_sample_rate = SAMPLING_RATE

        self.muse_samples_needed = int(
            self.window_seconds * MUSE_SAMPLE_RATE
        )
        self.model_samples_needed = int(
            self.window_seconds * self.model_sample_rate
        )

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._artifact: dict[str, Any] | None = None

        self.running = False
        self.last_error = ""
        self.latest_result: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []

    def start(self) -> None:
        """Start realtime detection in a background thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop realtime detection."""
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=3)

    def _load_artifact_once(self) -> dict[str, Any]:
        """Load the trained model once and reuse it."""
        if self._artifact is None:
            self._artifact = load_model(self.model_file)

        return self._artifact

    def _run(self) -> None:
        """Continuously read recent Muse windows and predict emotions."""
        self.running = True
        self.last_error = ""

        try:
            self._load_artifact_once()

            while not self._stop_event.is_set():
                raw_window = self.receiver.latest_window(
                    self.muse_samples_needed,
                    MUSE_CHANNEL_ORDER_FOR_MODEL,
                )

                if raw_window is not None:
                    result = self.predict(raw_window)

                    with self._lock:
                        self.latest_result = result
                        self.history.append(result)
                        self.history = self.history[-20:]

                time.sleep(self.step_seconds)

        except Exception as error:
            self.last_error = str(error)

        finally:
            self.running = False

    def preprocess_live_window(
        self,
        muse_window: np.ndarray,
    ) -> np.ndarray:
        """Convert Muse 256 Hz data to the 200 Hz model format."""
        window = np.asarray(muse_window, dtype=np.float64)

        window = replace_bad_values(window)
        window = signal.resample(
            window,
            self.model_samples_needed,
            axis=1,
        )
        window = signal.detrend(
            window,
            axis=1,
            type="constant",
        )
        window = bandpass_filter(
            window,
            self.model_sample_rate,
        )

        return window.astype(np.float32)

    def predict(
        self,
        muse_window: np.ndarray,
    ) -> dict[str, Any]:
        """Predict one emotion from a Muse EEG window."""
        artifact = self._load_artifact_once()
        model = artifact["model"]

        label_names = {
            int(key): value
            for key, value in artifact.get(
                "label_names",
                LABEL_NAMES,
            ).items()
        }

        model_window = self.preprocess_live_window(
            muse_window
        )

        features, _ = extract_feature_vector(
            model_window,
            sampling_rate=self.model_sample_rate,
            channel_names=SELECTED_SEED_CHANNELS,
        )

        features = features.reshape(1, -1)

        predicted_class = int(
            model.predict(features)[0]
        )

        probabilities = (
            model.predict_proba(features)[0]
            if hasattr(model, "predict_proba")
            else None
        )

        return {
            "timestamp": time.time(),
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
            "model_name": artifact.get(
                "model_name",
                "unknown",
            ),
            "window_seconds": self.window_seconds,
        }

    def status(self) -> dict[str, Any]:
        """Return current detector status for the API."""
        with self._lock:
            latest = (
                dict(self.latest_result)
                if self.latest_result
                else None
            )
            history = list(self.history)

        return {
            "running": self.running,
            "last_error": self.last_error,
            "model_file": str(self.model_file),
            "window_seconds": self.window_seconds,
            "muse_samples_needed": self.muse_samples_needed,
            "latest_result": latest,
            "history": history,
        }


def replace_bad_values(
    window: np.ndarray,
) -> np.ndarray:
    """Replace NaN/Inf values with the channel median."""
    cleaned = window.copy()

    for channel_index in range(cleaned.shape[0]):
        channel = cleaned[channel_index]
        good = np.isfinite(channel)

        fill_value = (
            float(np.median(channel[good]))
            if good.any()
            else 0.0
        )

        channel[~good] = fill_value

    return cleaned


def bandpass_filter(
    window: np.ndarray,
    sampling_rate: int,
) -> np.ndarray:
    """Apply the same bandpass filter used during training."""
    nyquist = sampling_rate / 2.0

    low = FILTER_LOW_HZ / nyquist
    high = FILTER_HIGH_HZ / nyquist

    sos = signal.butter(
        FILTER_ORDER,
        [low, high],
        btype="bandpass",
        output="sos",
    )

    return signal.sosfiltfilt(
        sos,
        window,
        axis=1,
    )


def probability_map(
    model: Any,
    probabilities: np.ndarray | None,
    label_names: dict[int, str],
) -> dict[str, float] | None:
    """Convert model probabilities into an emotion-name dictionary."""
    if probabilities is None:
        return None

    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        final_step = list(
            model.named_steps.values()
        )[-1]
        classes = getattr(
            final_step,
            "classes_",
            None,
        )

    if classes is None:
        classes = np.arange(len(probabilities))

    return {
        label_names.get(
            int(class_id),
            str(class_id),
        ): float(probabilities[index])
        for index, class_id in enumerate(classes)
    }
