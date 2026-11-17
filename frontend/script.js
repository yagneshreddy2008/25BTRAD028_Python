/**
 * SignBridge AI – Frontend Application Controller
 * Handles webcam stream, MediaPipe Hands landmark extraction, WebSocket communication,
 * UI updates, sentence construction, and Speech Synthesis.
 */

// Application State
const AppState = {
  isStreaming: false,
  isTranslating: true,
  camera: null,
  handsDetector: null,
  ws: null,
  wsUrl: `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.hostname || "127.0.0.1"}:8000/ws/landmarks`,
  
  // Telemetry
  frameCount: 0,
  lastFpsTimestamp: performance.now(),
  currentFps: 0,
  lastPingTimestamp: 0,
  currentLatency: 0,

  // Settings
  showLandmarks: true,
  isMirrored: true,
  autoSpeak: false,
  selectedVoice: null,
  speechRate: 1.0,

  // Translation data
  sentenceHistory: []
};

// DOM Elements
const DOM = {
  // Navigation & Status
  wsIndicator: document.getElementById("ws-indicator"),
  wsStatusText: document.getElementById("ws-status-text"),

  // Webcam & Canvas
  video: document.getElementById("webcam-video"),
  canvas: document.getElementById("landmark-canvas"),
  placeholder: document.getElementById("webcam-placeholder"),
  btnStartCamera: document.getElementById("btn-start-camera"),
  btnStopCamera: document.getElementById("btn-stop-camera"),
  btnStartCameraPrompt: document.getElementById("btn-start-camera-prompt"),
  toggleLandmarks: document.getElementById("toggle-landmarks"),
  toggleMirror: document.getElementById("toggle-mirror"),

  // Telemetry & Badges
  fpsDisplay: document.getElementById("fps-display"),
  latencyDisplay: document.getElementById("latency-display"),
  cameraStatusPill: document.getElementById("camera-status-pill"),
  handsBadge: document.getElementById("hands-badge"),
  diagLeftHand: document.getElementById("diag-left-hand"),
  diagRightHand: document.getElementById("diag-right-hand"),
  diagFeatures: document.getElementById("diag-features"),

  // Holding Progress
  candidateProgressContainer: document.getElementById("candidate-progress-container"),
  candidateName: document.getElementById("candidate-name"),
  candidatePct: document.getElementById("candidate-pct"),
  candidateProgressBar: document.getElementById("candidate-progress-bar"),

  // Recognition Display
  liveWordDisplay: document.getElementById("live-word-display"),
  confidencePct: document.getElementById("confidence-pct"),
  confidenceBar: document.getElementById("confidence-bar"),
  lastConfirmedPill: document.getElementById("last-confirmed-pill"),
  engineStatus: document.getElementById("engine-status"),

  // Sentence Elements
  sentenceOutput: document.getElementById("sentence-output"),
  wordCountBadge: document.getElementById("word-count-badge"),
  btnSpeak: document.getElementById("btn-speak"),
  btnCopy: document.getElementById("btn-copy"),
  btnDeleteLast: document.getElementById("btn-delete-last"),
  btnClear: document.getElementById("btn-clear"),

  // Speech Drawer
  ttsVoiceSelect: document.getElementById("tts-voice-select"),
  ttsRateSlider: document.getElementById("tts-rate-slider"),
  ttsRateVal: document.getElementById("tts-rate-val"),
  ttsAutoSpeak: document.getElementById("tts-auto-speak"),

  // Session Controls
  btnPauseTrans: document.getElementById("btn-pause-trans"),
  btnResumeTrans: document.getElementById("btn-resume-trans"),
  btnResetSession: document.getElementById("btn-reset-session"),

  // History & Toast
  historyList: document.getElementById("history-list"),
  historyEmpty: document.getElementById("history-empty"),
  btnClearHistory: document.getElementById("btn-clear-history"),
  toastContainer: document.getElementById("toast-container")
};

const canvasCtx = DOM.canvas.getContext("2d");

/* -------------------------------------------------------------
   Initialize Application
   ------------------------------------------------------------- */
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  initSpeechSynthesis();
  initWebSocket();
  initMediaPipe();
});

/* -------------------------------------------------------------
   Event Listeners
   ------------------------------------------------------------- */
function setupEventListeners() {
  DOM.btnStartCamera.addEventListener("click", startWebcam);
  DOM.btnStartCameraPrompt.addEventListener("click", startWebcam);
  DOM.btnStopCamera.addEventListener("click", stopWebcam);

  DOM.toggleLandmarks.addEventListener("change", (e) => {
    AppState.showLandmarks = e.target.checked;
    if (!AppState.showLandmarks) {
      canvasCtx.clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);
    }
  });

  DOM.toggleMirror.addEventListener("change", (e) => {
    AppState.isMirrored = e.target.checked;
    DOM.video.classList.toggle("mirrored", AppState.isMirrored);
    DOM.canvas.classList.toggle("mirrored", AppState.isMirrored);
  });

  // Translation Actions
  DOM.btnSpeak.addEventListener("click", speakCurrentSentence);
  DOM.btnCopy.addEventListener("click", copySentenceToClipboard);
  DOM.btnDeleteLast.addEventListener("click", sendDeleteLastWord);
  DOM.btnClear.addEventListener("click", sendClearSentence);

  // Session Controls
  DOM.btnPauseTrans.addEventListener("click", () => setTranslationPaused(true));
  DOM.btnResumeTrans.addEventListener("click", () => setTranslationPaused(false));
  DOM.btnResetSession.addEventListener("click", resetTranslationSession);
  DOM.btnClearHistory.addEventListener("click", clearHistory);

  // Speech settings
  DOM.ttsVoiceSelect.addEventListener("change", (e) => {
    AppState.selectedVoice = e.target.value;
  });
  DOM.ttsRateSlider.addEventListener("input", (e) => {
    AppState.speechRate = parseFloat(e.target.value);
    DOM.ttsRateVal.textContent = `${AppState.speechRate.toFixed(1)}x`;
  });
  DOM.ttsAutoSpeak.addEventListener("change", (e) => {
    AppState.autoSpeak = e.target.checked;
  });

  // Initial mirror setup
  DOM.video.classList.toggle("mirrored", AppState.isMirrored);
  DOM.canvas.classList.toggle("mirrored", AppState.isMirrored);
}

/* -------------------------------------------------------------
   WebSocket Communication
   ------------------------------------------------------------- */
let wsReconnectTimer = null;

function initWebSocket() {
  if (AppState.ws && (AppState.ws.readyState === WebSocket.OPEN || AppState.ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  updateWsStatus("connecting");
  try {
    AppState.ws = new WebSocket(AppState.wsUrl);

    AppState.ws.onopen = () => {
      console.log("WebSocket connected to SignBridge backend.");
      updateWsStatus("connected");
      clearTimeout(wsReconnectTimer);
    };

    AppState.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
      } catch (err) {
        console.error("Error parsing WebSocket message:", err);
      }
    };

    AppState.ws.onclose = () => {
      updateWsStatus("disconnected");
      scheduleWsReconnect();
    };

    AppState.ws.onerror = (err) => {
      console.warn("WebSocket encounter:", err);
      updateWsStatus("disconnected");
    };
  } catch (err) {
    console.error("WebSocket init failed:", err);
    updateWsStatus("disconnected");
    scheduleWsReconnect();
  }
}

function scheduleWsReconnect() {
  clearTimeout(wsReconnectTimer);
  wsReconnectTimer = setTimeout(() => {
    console.log("Attempting WebSocket reconnection...");
    initWebSocket();
  }, 2500);
}

function updateWsStatus(status) {
  if (status === "connected") {
    DOM.wsIndicator.className = "status-pill status-connected";
    DOM.wsStatusText.textContent = "AI Connected";
    DOM.engineStatus.className = "badge badge-success";
    DOM.engineStatus.textContent = "Engine Active";
  } else if (status === "connecting") {
    DOM.wsIndicator.className = "status-pill status-neutral";
    DOM.wsStatusText.textContent = "Connecting...";
    DOM.engineStatus.className = "badge badge-neutral";
    DOM.engineStatus.textContent = "Connecting...";
  } else {
    DOM.wsIndicator.className = "status-pill status-disconnected";
    DOM.wsStatusText.textContent = "AI Offline";
    DOM.engineStatus.className = "badge badge-warning";
    DOM.engineStatus.textContent = "Engine Offline";
  }
}

function sendWsMessage(payload) {
  if (AppState.ws && AppState.ws.readyState === WebSocket.OPEN) {
    AppState.ws.send(JSON.stringify(payload));
  }
}

/* -------------------------------------------------------------
   Handle Server Prediction & Sentence Updates
   ------------------------------------------------------------- */
function handleServerMessage(data) {
  // Latency calculation
  if (data.client_timestamp) {
    AppState.currentLatency = Math.max(1, Math.round(performance.now() - data.client_timestamp));
    DOM.latencyDisplay.textContent = AppState.currentLatency;
  }

  // Handle Real-Time Prediction Updates
  if (data.type === "prediction") {
    const word = data.predicted_word || "WAITING FOR GESTURE";
    const conf = Math.round((data.confidence || 0) * 100);
    const candidate = data.candidate_word;
    const progress = Math.min(100, Math.round((data.candidate_progress || 0) * 100));

    // Update Live Hero Box
    DOM.liveWordDisplay.textContent = word !== "NO_HANDS" ? word : "SHOW HAND GESTURE";
    DOM.confidencePct.textContent = `${conf}%`;
    DOM.confidenceBar.style.width = `${conf}%`;

    // Candidate progress bar (hold indicator)
    if (candidate && candidate !== "NEUTRAL" && candidate !== "NO_HANDS") {
      DOM.candidateProgressContainer.style.display = "block";
      DOM.candidateName.textContent = candidate;
      const displayPct = progress > 0 ? progress : Math.min(100, Math.round(conf * 0.8));
      DOM.candidatePct.textContent = `${displayPct}%`;
      DOM.candidateProgressBar.style.width = `${displayPct}%`;
    } else {
      DOM.candidateProgressContainer.style.display = "none";
    }

    if (data.last_confirmed) {
      DOM.lastConfirmedPill.textContent = data.last_confirmed;
    }
  }

  // Handle Confirmed Word Event
  if (data.type === "word_confirmed") {
    triggerWordConfirmedAnimation(data.word);
    showToast(`Confirmed sign: "${data.word}"`);
  }

  // Handle Full Sentence State Updates
  if (data.type === "sentence_update" || data.sentence_state) {
    const state = data.sentence_state || data;
    updateSentenceUI(state);
  }
}

function triggerWordConfirmedAnimation(word) {
  DOM.liveWordDisplay.classList.add("confirmed-flash");
  setTimeout(() => {
    DOM.liveWordDisplay.classList.remove("confirmed-flash");
  }, 400);
}

function updateSentenceUI(state) {
  const formatted = state.formatted_sentence || "";
  const wordCount = state.word_count !== undefined ? state.word_count : (state.words ? state.words.length : 0);

  DOM.wordCountBadge.textContent = `${wordCount} Word${wordCount === 1 ? "" : "s"}`;

  if (formatted.trim()) {
    DOM.sentenceOutput.innerHTML = `<span class="active-sentence">${escapeHtml(formatted)}</span>`;
  } else {
    DOM.sentenceOutput.innerHTML = `<span class="placeholder-text">Your translated sentence will be assembled here with smart grammar, automatic capitalization, and punctuation...</span>`;
  }

  if (state.last_confirmed_word) {
    DOM.lastConfirmedPill.textContent = state.last_confirmed_word;
  }
}

/* -------------------------------------------------------------
   Webcam Stream & MediaPipe Integration
   ------------------------------------------------------------- */
async function startWebcam() {
  if (AppState.isStreaming) return;

  DOM.cameraStatusPill.textContent = "Requesting Camera...";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 30 }
      },
      audio: false
    });

    DOM.video.srcObject = stream;
    await DOM.video.play();

    AppState.isStreaming = true;
    DOM.placeholder.style.display = "none";
    DOM.btnStartCamera.disabled = true;
    DOM.btnStopCamera.disabled = false;
    DOM.cameraStatusPill.textContent = "Camera Live (30 FPS)";

    // Match canvas dimensions to video
    DOM.video.addEventListener("loadedmetadata", () => {
      DOM.canvas.width = DOM.video.videoWidth || 640;
      DOM.canvas.height = DOM.video.videoHeight || 480;
    });

    // Start sending frames to MediaPipe Hands
    startMediaPipeCamera();
    showToast("Webcam connected successfully.");
  } catch (err) {
    console.error("Camera access error:", err);
    DOM.cameraStatusPill.textContent = "Camera Error";
    let msg = "Could not access webcam. Please check permissions.";
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      msg = "Camera permission was denied. Please allow camera access in browser.";
    } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      msg = "No webcam device found on this system.";
    }
    showToast(msg, "error");
  }
}

function stopWebcam() {
  if (!AppState.isStreaming) return;

  if (DOM.video.srcObject) {
    DOM.video.srcObject.getTracks().forEach((track) => track.stop());
    DOM.video.srcObject = null;
  }

  AppState.isStreaming = false;
  DOM.placeholder.style.display = "flex";
  DOM.btnStartCamera.disabled = false;
  DOM.btnStopCamera.disabled = true;
  DOM.cameraStatusPill.textContent = "Camera Standby";

  canvasCtx.clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);
  updateHandsTelemetry(false, false);
  showToast("Camera stopped.");
}

function initMediaPipe() {
  if (typeof Hands === "undefined") {
    console.warn("MediaPipe Hands library not loaded from CDN.");
    return;
  }

  AppState.handsDetector = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  AppState.handsDetector.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  AppState.handsDetector.onResults(onMediaPipeResults);
}

function startMediaPipeCamera() {
  if (!AppState.handsDetector) {
    initMediaPipe();
  }

  async function processVideoFrame() {
    if (!AppState.isStreaming) return;

    if (DOM.video.readyState >= 2) {
      await AppState.handsDetector.send({ image: DOM.video });
      calculateFps();
    }
    requestAnimationFrame(processVideoFrame);
  }

  requestAnimationFrame(processVideoFrame);
}

/* -------------------------------------------------------------
   MediaPipe Results & Landmark Transmission
   ------------------------------------------------------------- */
function onMediaPipeResults(results) {
  // Sync canvas size
  if (DOM.canvas.width !== DOM.video.videoWidth && DOM.video.videoWidth > 0) {
    DOM.canvas.width = DOM.video.videoWidth;
    DOM.canvas.height = DOM.video.videoHeight;
  }

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, DOM.canvas.width, DOM.canvas.height);

  let leftHandLandmarks = null;
  let rightHandLandmarks = null;

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const landmarks = results.multiHandLandmarks[i];
      let handednessLabel = "Right"; // default

      if (results.multiHandedness && results.multiHandedness[i]) {
        handednessLabel = results.multiHandedness[i].label; // "Left" or "Right"
      }

      // Assign to respective hand slot
      if (handednessLabel === "Left") {
        leftHandLandmarks = landmarks;
      } else {
        rightHandLandmarks = landmarks;
      }

      // Render skeletal lines & landmark dots
      if (AppState.showLandmarks) {
        const isLeft = handednessLabel === "Left";
        const connectorColor = isLeft ? "#06b6d4" : "#a855f7"; // Cyan for left, Violet for right
        const landmarkColor = isLeft ? "#22d3ee" : "#c084fc";

        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
          color: connectorColor,
          lineWidth: 3
        });
        drawLandmarks(canvasCtx, landmarks, {
          color: landmarkColor,
          fillColor: "#ffffff",
          lineWidth: 1.5,
          radius: 3.5
        });
      }
    }
  }

  canvasCtx.restore();

  // Update UI telemetry badges
  const hasLeft = leftHandLandmarks !== null;
  const hasRight = rightHandLandmarks !== null;
  updateHandsTelemetry(hasLeft, hasRight);

  // Send landmarks to backend via WebSocket if translating
  if (AppState.isStreaming && AppState.isTranslating) {
    sendLandmarksToBackend(leftHandLandmarks, rightHandLandmarks);
  }
}

function updateHandsTelemetry(hasLeft, hasRight) {
  DOM.diagLeftHand.textContent = hasLeft ? "Detected (21 pts)" : "Inactive";
  DOM.diagRightHand.textContent = hasRight ? "Detected (21 pts)" : "Inactive";
  DOM.diagLeftHand.style.color = hasLeft ? "var(--accent-cyan)" : "inherit";
  DOM.diagRightHand.style.color = hasRight ? "#c084fc" : "inherit";

  if (hasLeft && hasRight) {
    DOM.handsBadge.className = "badge badge-success";
    DOM.handsBadge.textContent = "Both Hands (42 Pts)";
  } else if (hasLeft) {
    DOM.handsBadge.className = "badge badge-info";
    DOM.handsBadge.textContent = "Left Hand Active";
  } else if (hasRight) {
    DOM.handsBadge.className = "badge badge-info";
    DOM.handsBadge.textContent = "Right Hand Active";
  } else {
    DOM.handsBadge.className = "badge badge-neutral";
    DOM.handsBadge.textContent = "No Hands Detected";
  }
}

function sendLandmarksToBackend(leftLandmarks, rightLandmarks) {
  if (!AppState.ws || AppState.ws.readyState !== WebSocket.OPEN) return;

  // Extract clean [{x, y, z}, ...] or null
  const cleanLeft = leftLandmarks
    ? leftLandmarks.map((pt) => ({ x: pt.x, y: pt.y, z: pt.z }))
    : null;
  const cleanRight = rightLandmarks
    ? rightLandmarks.map((pt) => ({ x: pt.x, y: pt.y, z: pt.z }))
    : null;

  AppState.lastPingTimestamp = performance.now();

  sendWsMessage({
    type: "landmarks",
    left_hand: cleanLeft,
    right_hand: cleanRight,
    client_timestamp: AppState.lastPingTimestamp
  });
}

function calculateFps() {
  AppState.frameCount++;
  const now = performance.now();
  const elapsed = now - AppState.lastFpsTimestamp;
  if (elapsed >= 1000) {
    AppState.currentFps = Math.round((AppState.frameCount * 1000) / elapsed);
    DOM.fpsDisplay.textContent = AppState.currentFps;
    AppState.frameCount = 0;
    AppState.lastFpsTimestamp = now;
  }
}

/* -------------------------------------------------------------
   Sentence Actions & Server Handlers
   ------------------------------------------------------------- */
function sendDeleteLastWord() {
  sendWsMessage({ type: "action", action: "delete_last" });
  showToast("Removed last word.");
}

function sendClearSentence() {
  // Save to history before clearing if not empty
  archiveCurrentSentence();
  sendWsMessage({ type: "action", action: "clear_sentence" });
  showToast("Sentence cleared.");
}

function setTranslationPaused(paused) {
  AppState.isTranslating = !paused;
  DOM.btnPauseTrans.style.display = paused ? "none" : "inline-flex";
  DOM.btnResumeTrans.style.display = paused ? "inline-flex" : "none";
  DOM.engineStatus.textContent = paused ? "Translation Paused" : "Engine Active";
  DOM.engineStatus.className = paused ? "badge badge-warning" : "badge badge-success";
  sendWsMessage({ type: "action", action: paused ? "pause" : "resume" });
  showToast(paused ? "Translation paused." : "Translation resumed.");
}

function resetTranslationSession() {
  archiveCurrentSentence();
  sendWsMessage({ type: "action", action: "reset_session" });
  DOM.liveWordDisplay.textContent = "WAITING FOR GESTURE";
  DOM.confidencePct.textContent = "0%";
  DOM.confidenceBar.style.width = "0%";
  DOM.lastConfirmedPill.textContent = "None";
  DOM.wordCountBadge.textContent = "0 Words";
  DOM.sentenceOutput.innerHTML = `<span class="placeholder-text">Your translated sentence will be assembled here with smart grammar, automatic capitalization, and punctuation...</span>`;
  showToast("Session reset.");
}

/* -------------------------------------------------------------
   Text to Speech (Web Speech API)
   ------------------------------------------------------------- */
let speechSynth = window.speechSynthesis;
let availableVoices = [];

function initSpeechSynthesis() {
  if (!speechSynth) {
    console.warn("SpeechSynthesis is not supported in this browser.");
    DOM.ttsVoiceSelect.disabled = true;
    return;
  }

  function populateVoices() {
    availableVoices = speechSynth.getVoices();
    DOM.ttsVoiceSelect.innerHTML = '<option value="">Default System Voice</option>';

    availableVoices.forEach((voice, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `${voice.name} (${voice.lang})${voice.default ? " — Default" : ""}`;
      DOM.ttsVoiceSelect.appendChild(opt);
    });
  }

  populateVoices();
  if (speechSynth.onvoiceschanged !== undefined) {
    speechSynth.onvoiceschanged = populateVoices;
  }
}

function speakCurrentSentence() {
  const text = DOM.sentenceOutput.textContent.trim();
  if (!text || text.startsWith("Your translated sentence")) {
    showToast("No sentence to speak.", "warning");
    return;
  }
  speakText(text);
}

function speakText(text) {
  if (!speechSynth) {
    showToast("Speech synthesis not supported.", "error");
    return;
  }

  speechSynth.cancel(); // Stop any pending speech
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = AppState.speechRate;
  utterance.pitch = 1.0;

  if (AppState.selectedVoice !== null && availableVoices[AppState.selectedVoice]) {
    utterance.voice = availableVoices[AppState.selectedVoice];
  }

  utterance.onstart = () => {
    DOM.btnSpeak.classList.add("btn-active");
  };
  utterance.onend = () => {
    DOM.btnSpeak.classList.remove("btn-active");
  };

  speechSynth.speak(utterance);
}

/* -------------------------------------------------------------
   Sentence History & Clipboard Utilities
   ------------------------------------------------------------- */
function archiveCurrentSentence() {
  const text = DOM.sentenceOutput.textContent.trim();
  if (!text || text.startsWith("Your translated sentence")) return;

  const entry = {
    text: text,
    time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  };

  AppState.sentenceHistory.unshift(entry);
  renderHistory();
}

function renderHistory() {
  if (AppState.sentenceHistory.length === 0) {
    DOM.historyEmpty.style.display = "block";
    DOM.historyList.innerHTML = "";
    DOM.historyList.appendChild(DOM.historyEmpty);
    return;
  }

  DOM.historyEmpty.style.display = "none";
  DOM.historyList.innerHTML = "";

  AppState.sentenceHistory.slice(0, 15).forEach((item) => {
    const el = document.createElement("div");
    el.className = "history-item";
    el.innerHTML = `
      <span class="history-text">${escapeHtml(item.text)}</span>
      <div class="history-meta">
        <span class="history-time">${item.time}</span>
        <button class="history-speak-btn" title="Speak this sentence">🔊</button>
      </div>
    `;

    el.querySelector(".history-speak-btn").addEventListener("click", () => {
      speakText(item.text);
    });

    DOM.historyList.appendChild(el);
  });
}

function clearHistory() {
  AppState.sentenceHistory = [];
  renderHistory();
  showToast("History cleared.");
}

async function copySentenceToClipboard() {
  const text = DOM.sentenceOutput.textContent.trim();
  if (!text || text.startsWith("Your translated sentence")) {
    showToast("Nothing to copy.", "warning");
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    showToast("Sentence copied to clipboard!");
  } catch (err) {
    // Fallback
    const textarea = document.createElement("textarea");
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
    showToast("Sentence copied to clipboard!");
  }
}

/* -------------------------------------------------------------
   UI Feedback & Toast Notifications
   ------------------------------------------------------------- */
function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = "toast";
  const icon = type === "error" ? "⚠️" : type === "warning" ? "⚡" : "✓";
  toast.innerHTML = `<span>${icon}</span> <span>${escapeHtml(message)}</span>`;

  DOM.toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    toast.style.transition = "all 0.2s ease";
    setTimeout(() => toast.remove(), 250);
  }, 2800);
}

function escapeHtml(str) {
  return str.replace(/[&<>'"]/g, 
    (tag) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      "'": "&#39;",
      '"': "&quot;"
    }[tag] || tag)
  );
}
