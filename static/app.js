const TARGET_SAMPLE_RATE = 16000;
// Batch mic audio into ~120 ms blocks before sending to drastically cut
// per-message overhead (hundreds of tiny WS frames/sec -> ~8/sec).
const CHUNK_MS = 120;

const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const partialEl = document.getElementById("partial");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const clearBtn = document.getElementById("clearBtn");

let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let workletNode = null;
let socket = null;
let captureChunks = [];
let capturedLength = 0;
let nativeChunkSize = 0;

function setStatus(text, connected) {
  statusEl.textContent = text;
  statusEl.className = connected ? "connected" : "disconnected";
}

// Linear-interpolation resampler: native mic rate -> 16 kHz mono.
function downsampleTo16k(float32Samples, inputSampleRate) {
  if (inputSampleRate === TARGET_SAMPLE_RATE) {
    return float32Samples;
  }
  const ratio = inputSampleRate / TARGET_SAMPLE_RATE;
  const outLength = Math.round(float32Samples.length / ratio);
  const result = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const srcPos = i * ratio;
    const srcIndex = Math.floor(srcPos);
    const frac = srcPos - srcIndex;
    const a = float32Samples[srcIndex] || 0;
    const b = float32Samples[srcIndex + 1] || a;
    result[i] = a + (b - a) * frac;
  }
  return result;
}

function floatTo16BitPCM(float32Samples) {
  const buffer = new ArrayBuffer(float32Samples.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < float32Samples.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Samples[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

async function start() {
  startBtn.disabled = true;

  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws/asr`);
  socket.binaryType = "arraybuffer";

  socket.onopen = () => setStatus("Connected - listening", true);
  socket.onclose = () => setStatus("Disconnected", false);
  socket.onerror = () => setStatus("Connection error", false);
  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.is_final) {
      if (data.text) {
        transcriptEl.textContent += (transcriptEl.textContent ? "\n" : "") + data.text;
      }
      partialEl.textContent = "";
    } else {
      partialEl.textContent = data.text;
    }
  };

  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new AudioContext();
  await audioContext.audioWorklet.addModule("/static/pcm-processor.js");

  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioContext, "pcm-capture-processor");

  captureChunks = [];
  capturedLength = 0;
  nativeChunkSize = Math.round((audioContext.sampleRate * CHUNK_MS) / 1000);

  workletNode.port.onmessage = (event) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    captureChunks.push(event.data);
    capturedLength += event.data.length;
    if (capturedLength < nativeChunkSize) return;
    flushCaptureBuffer();
  };

  sourceNode.connect(workletNode);
  stopBtn.disabled = false;
}

// Merge buffered native-rate samples, resample to 16 kHz and send one frame.
function flushCaptureBuffer() {
  if (capturedLength === 0 || !socket || socket.readyState !== WebSocket.OPEN) return;
  const merged = new Float32Array(capturedLength);
  let offset = 0;
  for (const chunk of captureChunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  captureChunks = [];
  capturedLength = 0;
  const resampled = downsampleTo16k(merged, audioContext.sampleRate);
  socket.send(floatTo16BitPCM(resampled));
}

function stop() {
  stopBtn.disabled = true;

  flushCaptureBuffer();

  if (workletNode) {
    workletNode.disconnect();
    workletNode = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (socket) {
    socket.close();
    socket = null;
  }

  partialEl.textContent = "";
  setStatus("Disconnected", false);
  startBtn.disabled = false;
}

function clearTranscript() {
  transcriptEl.textContent = "";
  partialEl.textContent = "";
}

startBtn.addEventListener("click", () => start().catch((err) => {
  console.error(err);
  setStatus(`Error: ${err.message}`, false);
  startBtn.disabled = false;
}));
stopBtn.addEventListener("click", stop);
clearBtn.addEventListener("click", clearTranscript);

// ---------------- Tabs ----------------
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.tab).classList.add("active");
    if (btn.dataset.tab === "history") {
      loadHistory();
    }
  });
});

// ---------------- History / conversation storage ----------------
const dateFilter = document.getElementById("dateFilter");
const searchFilter = document.getElementById("searchFilter");
const historyListEl = document.getElementById("historyList");
const detailEl = document.getElementById("conversationDetail");
const titleInput = document.getElementById("titleInput");
const detailTranscript = document.getElementById("detailTranscript");
const detailAudio = document.getElementById("detailAudio");

let activeConversationId = null;

async function loadHistory() {
  const params = new URLSearchParams();
  if (dateFilter.value) params.set("date", dateFilter.value);
  if (searchFilter.value) params.set("search", searchFilter.value);

  const res = await fetch(`/api/conversations?${params.toString()}`);
  const items = await res.json();
  renderHistoryList(items);
}

function renderHistoryList(items) {
  historyListEl.innerHTML = "";
  if (items.length === 0) {
    const li = document.createElement("li");
    li.textContent = "Không có cuộc hội thoại nào.";
    historyListEl.appendChild(li);
    return;
  }
  for (const item of items) {
    const li = document.createElement("li");
    li.dataset.id = item.id;
    const title = item.title || "(Chưa có nội dung)";
    li.innerHTML = `<div>${title}</div><div class="conv-meta">${item.date} · ${item.started_at.slice(11, 16)} · ${item.segment_count} đoạn</div>`;
    li.addEventListener("click", () => openConversation(item.id, li));
    historyListEl.appendChild(li);
  }
}

async function openConversation(id, liEl) {
  document.querySelectorAll("#historyList li").forEach((el) => el.classList.remove("active"));
  if (liEl) liEl.classList.add("active");

  const res = await fetch(`/api/conversations/${id}`);
  if (!res.ok) return;
  const conv = await res.json();
  activeConversationId = id;

  titleInput.value = conv.title || "";
  detailTranscript.value = conv.segments.map((s) => s.text).join("\n");
  if (conv.audio_path) {
    detailAudio.src = `/api/conversations/${id}/audio`;
    detailAudio.style.display = "block";
  } else {
    detailAudio.style.display = "none";
    detailAudio.removeAttribute("src");
  }
  detailEl.style.display = "block";
}

document.getElementById("refreshHistoryBtn").addEventListener("click", loadHistory);
document.getElementById("clearFilterBtn").addEventListener("click", () => {
  dateFilter.value = "";
  searchFilter.value = "";
  loadHistory();
});
dateFilter.addEventListener("change", loadHistory);
searchFilter.addEventListener("keydown", (e) => {
  if (e.key === "Enter") loadHistory();
});

document.getElementById("saveTitleBtn").addEventListener("click", async () => {
  if (!activeConversationId) return;
  await fetch(`/api/conversations/${activeConversationId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: titleInput.value }),
  });
  loadHistory();
});

document.getElementById("exportBtn").addEventListener("click", () => {
  if (!activeConversationId) return;
  window.location.href = `/api/conversations/${activeConversationId}/export`;
});

document.getElementById("deleteConvBtn").addEventListener("click", async () => {
  if (!activeConversationId) return;
  if (!confirm("Xóa cuộc hội thoại này?")) return;
  await fetch(`/api/conversations/${activeConversationId}`, { method: "DELETE" });
  activeConversationId = null;
  detailEl.style.display = "none";
  loadHistory();
});

