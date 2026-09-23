const addressInput = document.getElementById("addressInput");
const connectBtn = document.getElementById("connectBtn");
const disconnectBtn = document.getElementById("disconnectBtn");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusText = document.getElementById("statusText");
const modelName = document.getElementById("modelName");
const emotionLabel = document.getElementById("emotionLabel");
const probabilities = document.getElementById("probabilities");
const connectionState = document.getElementById("connectionState");
const detectionState = document.getElementById("detectionState");
const windowInfo = document.getElementById("windowInfo");
const bufferInfo = document.getElementById("bufferInfo");
const errorText = document.getElementById("errorText");
const historyList = document.getElementById("historyList");
const eegCanvas = document.getElementById("eegCanvas");
const waveStatus = document.getElementById("waveStatus");

const waveColors = {
  TP9: "#1769aa",
  AF7: "#16866f",
  AF8: "#c84b55",
  TP10: "#b06a10",
};

let latestWaveData = null;
let waveRequestRunning = false;

async function postJson(url, payload = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.message || "Request failed.");
  }
  return data;
}

function formatProbability(value) {
  return `${Math.round(value * 100)}%`;
}

function renderProbabilities(probabilityMap) {
  probabilities.innerHTML = "";
  if (!probabilityMap) {
    probabilities.textContent = "No probabilities available.";
    return;
  }

  Object.entries(probabilityMap).forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span>${label}</span>
      <span class="prob-track"><span class="prob-fill" style="width: ${Math.max(0, Math.min(100, value * 100))}%"></span></span>
      <span>${formatProbability(value)}</span>
    `;
    probabilities.appendChild(row);
  });
}

function renderHistory(history) {
  historyList.innerHTML = "";
  const rows = [...history].reverse().slice(0, 8);
  rows.forEach((item) => {
    const li = document.createElement("li");
    const time = new Date(item.timestamp * 1000).toLocaleTimeString();
    li.textContent = `${time} - ${item.predicted_label}`;
    historyList.appendChild(li);
  });
}

function canvasContext() {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, eegCanvas.clientWidth);
  const height = Math.max(1, eegCanvas.clientHeight);
  const pixelWidth = Math.round(width * ratio);
  const pixelHeight = Math.round(height * ratio);

  if (eegCanvas.width !== pixelWidth || eegCanvas.height !== pixelHeight) {
    eegCanvas.width = pixelWidth;
    eegCanvas.height = pixelHeight;
  }

  const context = eegCanvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width, height };
}

function drawWaveGrid(context, width, height, margins, channelCount) {
  const plotWidth = width - margins.left - margins.right;
  const plotHeight = height - margins.top - margins.bottom;

  context.strokeStyle = "#e5eaee";
  context.lineWidth = 1;
  for (let index = 0; index <= 8; index += 1) {
    const x = margins.left + (plotWidth * index) / 8;
    context.beginPath();
    context.moveTo(x, margins.top);
    context.lineTo(x, margins.top + plotHeight);
    context.stroke();
  }

  for (let index = 0; index <= channelCount; index += 1) {
    const y = margins.top + (plotHeight * index) / channelCount;
    context.beginPath();
    context.moveTo(margins.left, y);
    context.lineTo(width - margins.right, y);
    context.stroke();
  }
}

function robustAmplitudeScale(channelValues) {
  const magnitudes = [];
  channelValues.forEach((values) => {
    if (!values.length) return;
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    values.forEach((value) => magnitudes.push(Math.abs(value - mean)));
  });
  if (!magnitudes.length) return 1;
  magnitudes.sort((a, b) => a - b);
  return Math.max(1, magnitudes[Math.floor((magnitudes.length - 1) * 0.95)]);
}

function drawWaves(data) {
  latestWaveData = data;
  const { context, width, height } = canvasContext();
  const margins = { left: 58, right: 18, top: 12, bottom: 26 };
  const order = data?.channel_order || ["TP9", "AF7", "AF8", "TP10"];
  const channelValues = order.map((channel) => data?.channels?.[channel] || []);

  context.clearRect(0, 0, width, height);
  context.fillStyle = "#fbfcfd";
  context.fillRect(0, 0, width, height);
  drawWaveGrid(context, width, height, margins, order.length);

  const plotWidth = width - margins.left - margins.right;
  const plotHeight = height - margins.top - margins.bottom;
  const laneHeight = plotHeight / order.length;

  context.font = "12px Arial";
  context.textBaseline = "middle";
  order.forEach((channel, index) => {
    const baseline = margins.top + laneHeight * (index + 0.5);
    context.fillStyle = waveColors[channel] || "#42515f";
    context.fillText(channel, 12, baseline);

    context.strokeStyle = "#cfd8df";
    context.beginPath();
    context.moveTo(margins.left, baseline);
    context.lineTo(width - margins.right, baseline);
    context.stroke();
  });

  if (!data || data.sample_count === 0) {
    context.fillStyle = "#657481";
    context.textAlign = "center";
    context.fillText("Connect the Muse headset to display live EEG waves.", width / 2, height / 2);
    context.textAlign = "left";
    waveStatus.textContent = "Waiting for EEG samples";
    return;
  }

  const scale = robustAmplitudeScale(channelValues);
  order.forEach((channel, index) => {
    const values = channelValues[index];
    if (values.length < 2) return;
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const baseline = margins.top + laneHeight * (index + 0.5);
    const amplitude = laneHeight * 0.38;

    context.strokeStyle = waveColors[channel] || "#42515f";
    context.lineWidth = 1.35;
    context.beginPath();
    values.forEach((value, pointIndex) => {
      const x = margins.left + (plotWidth * pointIndex) / (values.length - 1);
      const normalized = Math.max(-1, Math.min(1, (value - mean) / scale));
      const y = baseline - normalized * amplitude;
      if (pointIndex === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  });

  context.fillStyle = "#657481";
  context.textBaseline = "alphabetic";
  context.fillText(`-${data.duration_seconds.toFixed(1)} s`, margins.left, height - 7);
  context.textAlign = "right";
  context.fillText("now", width - margins.right, height - 7);
  context.textAlign = "left";
  waveStatus.textContent = data.connected
    ? `${data.point_count} display points/channel | ${data.sample_rate} Hz source`
    : "Showing the last buffered samples";
}

function renderStatus(data) {
  const receiver = data.receiver;
  const detector = data.detector;
  const connectedText = receiver.connected ? "connected" : receiver.running ? "connecting" : "idle";
  const detectionText = detector.running ? "running" : "stopped";
  const latest = detector.latest_result;

  statusText.textContent = `Device ${connectedText}; detection ${detectionText}.`;
  connectionState.textContent = connectedText;
  detectionState.textContent = detectionText;
  modelName.textContent = latest ? `model: ${latest.model_name}` : "model: waiting";
  windowInfo.textContent = `${detector.window_seconds}s, needs ${detector.muse_samples_needed} Muse samples/channel`;

  const buffers = receiver.buffer_lengths || {};
  bufferInfo.textContent = Object.entries(buffers)
    .map(([channel, count]) => `${channel}:${count}`)
    .join("  ");

  if (latest) {
    emotionLabel.textContent = latest.predicted_label;
    renderProbabilities(latest.probabilities);
  }

  renderHistory(detector.history || []);

  const errors = [receiver.last_error, detector.last_error].filter(Boolean);
  errorText.textContent = errors.join(" | ");

  connectBtn.disabled = receiver.running;
  disconnectBtn.disabled = !receiver.running;
  startBtn.disabled = !receiver.running || detector.running;
  stopBtn.disabled = !detector.running;
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status");
    renderStatus(await response.json());
  } catch (error) {
    errorText.textContent = error.message;
  }
}

async function refreshWaves() {
  if (waveRequestRunning) return;
  waveRequestRunning = true;
  try {
    const response = await fetch("/api/eeg-waves?seconds=4&points=420");
    if (!response.ok) throw new Error("Unable to load EEG waves.");
    drawWaves(await response.json());
  } catch (error) {
    waveStatus.textContent = error.message;
  } finally {
    waveRequestRunning = false;
  }
}

connectBtn.addEventListener("click", async () => {
  try {
    await postJson("/api/connect", { address: addressInput.value.trim() });
    await refreshStatus();
  } catch (error) {
    errorText.textContent = error.message;
  }
});

disconnectBtn.addEventListener("click", async () => {
  try {
    await postJson("/api/disconnect");
    await refreshStatus();
  } catch (error) {
    errorText.textContent = error.message;
  }
});

startBtn.addEventListener("click", async () => {
  try {
    await postJson("/api/start-detection");
    await refreshStatus();
  } catch (error) {
    errorText.textContent = error.message;
  }
});

stopBtn.addEventListener("click", async () => {
  try {
    await postJson("/api/stop-detection");
    await refreshStatus();
  } catch (error) {
    errorText.textContent = error.message;
  }
});

refreshStatus();
drawWaves(null);
refreshWaves();
setInterval(refreshStatus, 750);
setInterval(refreshWaves, 250);
window.addEventListener("resize", () => drawWaves(latestWaveData));
