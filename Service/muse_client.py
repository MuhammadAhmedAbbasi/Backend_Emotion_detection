"""Muse BLE receiver used by the realtime inference service."""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any

import numpy as np
from bleak import BleakClient


DEFAULT_ADDRESS = "00:55:DA:B6:32:68"

CONTROL_CHAR = "273e0001-4c4d-454d-96be-f03bac821358"
EEG_CHARS = {
    "TP9": "273e0003-4c4d-454d-96be-f03bac821358",
    "AF7": "273e0004-4c4d-454d-96be-f03bac821358",
    "AF8": "273e0005-4c4d-454d-96be-f03bac821358",
    "TP10": "273e0006-4c4d-454d-96be-f03bac821358",
}

MUSE_SAMPLE_RATE = 256
BUFFER_SECONDS = 30
BUFFER_SAMPLES = MUSE_SAMPLE_RATE * BUFFER_SECONDS


def encode_cmd(cmd: str) -> bytearray:
    return bytearray([len(cmd) + 1, *cmd.encode("ascii"), 0x0A])


def decode_12bit_samples(payload: bytes) -> list[int]:
    """Decode Muse's 18-byte payload into twelve 12-bit EEG samples."""
    if len(payload) != 18:
        return []

    samples: list[int] = []
    for i in range(0, 18, 3):
        x, y, z = payload[i], payload[i + 1], payload[i + 2]
        sample_1 = (x << 4) | (y >> 4)
        sample_2 = ((y & 0x0F) << 8) | z
        samples.extend([sample_1, sample_2])
    return samples


class MuseReceiver:
    """Small threaded BLE client that stores recent Muse EEG samples."""

    def __init__(self, address: str = DEFAULT_ADDRESS) -> None:
        self.address = address
        self.buffers = {
            channel: deque(maxlen=BUFFER_SAMPLES)
            for channel in EEG_CHARS
        }
        self.packet_counter = {channel: -1 for channel in EEG_CHARS}
        self.samples_per_channel = {channel: 0 for channel in EEG_CHARS}
        self.connected = False
        self.running = False
        self.last_error = ""
        self.started_at: float | None = None

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    def make_handler(self, channel_name: str):
        def handler(sender: Any, data: bytearray) -> None:
            if len(data) != 20:
                return

            packet_id = int.from_bytes(data[:2], byteorder="big")
            payload = data[2:]
            samples = decode_12bit_samples(payload)
            centered_samples = [sample - 2048 for sample in samples]

            with self._lock:
                self.buffers[channel_name].extend(centered_samples)
                self.packet_counter[channel_name] = packet_id
                self.samples_per_channel[channel_name] += len(centered_samples)

        return handler

    async def _send_cmd(self, client: BleakClient, cmd: str, wait_seconds: float = 0.3) -> None:
        await client.write_gatt_char(CONTROL_CHAR, encode_cmd(cmd))
        await asyncio.sleep(wait_seconds)

    async def _ble_main(self) -> None:
        try:
            async with BleakClient(self.address, timeout=20.0) as client:
                self.connected = bool(client.is_connected)
                self.running = True
                self.started_at = time.time()
                self.last_error = ""

                for channel_name, uuid_str in EEG_CHARS.items():
                    await client.start_notify(uuid_str, self.make_handler(channel_name))

                await self._send_cmd(client, "h")
                await self._send_cmd(client, "dc001")
                await self._send_cmd(client, "dc001")

                while not self._stop_event.is_set():
                    await asyncio.sleep(0.05)

                await self._send_cmd(client, "h")

                for uuid_str in EEG_CHARS.values():
                    try:
                        await client.stop_notify(uuid_str)
                    except Exception:
                        pass

        except Exception as exc:
            self.last_error = str(exc)
        finally:
            self.connected = False
            self.running = False

    def _thread_target(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._ble_main())
        self._loop.close()

    def start(self, address: str | None = None) -> None:
        if address:
            self.address = address
        if self._thread and self._thread.is_alive():
            return

        with self._lock:
            for channel in self.buffers:
                self.buffers[channel].clear()
                self.samples_per_channel[channel] = 0
                self.packet_counter[channel] = -1

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._thread_target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            buffer_lengths = {channel: len(buffer) for channel, buffer in self.buffers.items()}
            packet_counter = dict(self.packet_counter)
            samples_per_channel = dict(self.samples_per_channel)

        return {
            "address": self.address,
            "connected": self.connected,
            "running": self.running,
            "last_error": self.last_error,
            "started_at": self.started_at,
            "buffer_lengths": buffer_lengths,
            "packet_counter": packet_counter,
            "samples_per_channel": samples_per_channel,
        }

    def waveform_snapshot(
        self,
        samples_needed: int,
        max_points: int,
        channel_order: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return recent, downsampled EEG values for dashboard visualization."""
        order = channel_order or list(EEG_CHARS)
        with self._lock:
            available = min((len(self.buffers[channel]) for channel in order), default=0)
            sample_count = min(max(samples_needed, 0), available)
            recent = {
                channel: np.asarray(list(self.buffers[channel])[-sample_count:], dtype=np.float32)
                for channel in order
            }

        point_count = min(max(max_points, 1), sample_count)
        if point_count and point_count < sample_count:
            indices = np.linspace(0, sample_count - 1, point_count, dtype=np.int32)
            recent = {channel: values[indices] for channel, values in recent.items()}

        return {
            "connected": self.connected,
            "running": self.running,
            "sample_rate": MUSE_SAMPLE_RATE,
            "duration_seconds": sample_count / MUSE_SAMPLE_RATE,
            "sample_count": sample_count,
            "point_count": point_count,
            "unit": "centered ADC counts",
            "channel_order": order,
            "channels": {
                channel: values.astype(float).tolist()
                for channel, values in recent.items()
            },
        }

    def latest_window(self, samples_needed: int, channel_order: list[str]) -> np.ndarray | None:
        with self._lock:
            if any(len(self.buffers[channel]) < samples_needed for channel in channel_order):
                return None
            rows = [
                list(self.buffers[channel])[-samples_needed:]
                for channel in channel_order
            ]
        return np.asarray(rows, dtype=np.float32)
