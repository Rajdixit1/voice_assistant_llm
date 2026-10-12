// Browser side of the voice assistant: a continuous, phone-call style conversation.
// - While a call is active, every 32 ms mic frame (plus Silero's speech probability)
//   streams to the server. The server decides when the user started/stopped talking.
// - The server streams back raw PCM chunks, which are scheduled for gapless playback.
// - The server tells us to duck (user may be interrupting), unduck, or clear playback.

const VAD_VERSION = "0.0.24";
const ORT_VERSION = "1.22.0";
const FRAME_SAMPLES = 512;
const $ = (id) => document.getElementById(id);

const state = {
  ws: null,
  vad: null,
  inCall: false,
  callStartedAt: 0,
  timer: null,
  playCtx: null,
  gain: null,
  currentTurn: 0,
  sources: new Set(),
  nextTime: 0,
  serverDone: true,
  assistantEl: null,
  liveUserEl: null,
  hearingUntil: 0,
};

const LABELS = {
  offline: "Offline", idle: "Ready", listening: "Listening…", hearing: "Hearing you…",
  thinking: "Thinking…", speaking: "Speaking…",
};
function setStatus(s) {
  document.body.dataset.status = s;
  $("status-label").textContent = LABELS[s] || s;
}
const isSpeaking = () => state.sources.size > 0;
const idleStatus = () => (state.inCall ? "listening" : "idle");
const settle = () => { if (!isSpeaking() && state.serverDone) setStatus(idleStatus()); };

// WebSocket
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    setStatus(idleStatus());
    sendConfig();
    if (state.inCall) send({ type: "call_start", greeting: false });
  };
  ws.onclose = () => { setStatus("offline"); setTimeout(connect, 1500); };
  ws.onmessage = (ev) => (typeof ev.data === "string" ? onMessage(JSON.parse(ev.data)) : onAudio(ev.data));
  state.ws = ws;
}
const wsOpen = () => state.ws?.readyState === 1;
const send = (obj) => wsOpen() && state.ws.send(JSON.stringify(obj));
const sendConfig = () => send({ type: "config", language: $("language").value, barge_in: $("barge-in").checked });

function onMessage(msg) {
  switch (msg.type) {
    case "ready": {
      const voices = Object.entries(msg.tts).map(([l, v]) => `${l}: ${v}`).join(", ");
      $("models").textContent = `STT ${msg.stt} | LLM ${msg.llm} | TTS ${voices}`;
      break;
    }
    case "partial":
      if (!state.liveUserEl) state.liveUserEl = addMessage("user live", "");
      state.liveUserEl.querySelector(".text").textContent = msg.text;
      scrollLog();
      break;
    case "partial_cancel":
      dropLiveUser();
      break;
    case "turn_start":
      state.currentTurn = msg.turn;
      state.serverDone = false;
      state.assistantEl = null;
      setStatus("thinking");
      break;
    case "transcript":
      if (msg.turn !== state.currentTurn) return;
      if (state.liveUserEl) {
        const el = state.liveUserEl;
        state.liveUserEl = null;
        el.classList.remove("live");
        el.querySelector(".text").textContent = msg.text;
        el.prepend(langTag(msg.lang));
      } else {
        addMessage("user", msg.text, msg.lang);
      }
      break;
    case "token":
      if (msg.turn !== state.currentTurn) return;
      if (!state.assistantEl) state.assistantEl = addMessage("assistant", "");
      state.assistantEl.querySelector(".text").textContent += msg.text;
      scrollLog();
      break;
    case "metrics":
      if (msg.turn === state.currentTurn && state.assistantEl) showLatency(state.assistantEl, msg);
      break;
    case "done":
      if (msg.turn === state.currentTurn) { state.serverDone = true; settle(); }
      break;
    case "no_speech":
      if (msg.turn === state.currentTurn) { state.serverDone = true; dropLiveUser(); settle(); }
      break;
    case "duck":
      duck(true);
      break;
    case "unduck":
      duck(false);
      break;
    case "clear":
      stopPlayback();
      setStatus("hearing");
      break;
    case "reset_ok":
      $("log").innerHTML = "";
      break;
    case "error":
      addMessage("error", `Error: ${msg.message}`);
      state.serverDone = true;
      settle();
      break;
  }
}

// playback: [u32 turn][u32 sample rate][int16 PCM...]
function onAudio(buf) {
  const view = new DataView(buf);
  const turn = view.getUint32(0, true);
  const rate = view.getUint32(4, true);
  const ctx = state.playCtx;
  if (turn !== state.currentTurn || !ctx) return;
  const pcm = new Int16Array(buf, 8);
  const audio = ctx.createBuffer(1, pcm.length, rate);
  const data = audio.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) data[i] = pcm[i] / 32768;

  const src = ctx.createBufferSource();
  src.buffer = audio;
  src.connect(state.gain);
  const start = Math.max(ctx.currentTime + 0.01, state.nextTime);
  src.start(start);
  state.nextTime = start + audio.duration;
  state.sources.add(src);
  setStatus("speaking");
  src.onended = () => { state.sources.delete(src); settle(); };
}

function duck(on) {
  if (!state.gain) return;
  state.gain.gain.setTargetAtTime(on ? 0.15 : 1, state.playCtx.currentTime, 0.03);
}

function stopPlayback() {
  const wasActive = isSpeaking() || !state.serverDone;
  for (const src of state.sources) { src.onended = null; try { src.stop(); } catch (_) {} }
  state.sources.clear();
  state.nextTime = 0;
  state.currentTurn = -1;
  if (wasActive) state.assistantEl?.classList.add("cut");
  state.serverDone = true;
  duck(false);
}

function ensureAudio() {
  if (!state.playCtx) {
    state.playCtx = new AudioContext();
    state.gain = state.playCtx.createGain();
    state.gain.connect(state.playCtx.destination);
  }
  if (state.playCtx.state === "suspended") return state.playCtx.resume();
}

// microphone: stream every VAD frame to the server
const frameBuf = new ArrayBuffer(4 + FRAME_SAMPLES * 2);
const frameView = new DataView(frameBuf);
const framePcm = new Int16Array(frameBuf, 4);

function onFrame(probs, frame) {
  const p = probs.isSpeech;
  document.body.style.setProperty("--level", Math.min(1, p).toFixed(2));
  if (!state.inCall || !wsOpen() || frame.length !== FRAME_SAMPLES) return;
  frameView.setFloat32(0, p, true);
  for (let i = 0; i < FRAME_SAMPLES; i++) {
    const s = Math.max(-1, Math.min(1, frame[i]));
    framePcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  state.ws.send(frameBuf);

  // Local "hearing you" indicator; the server owns the real turn-taking.
  const now = performance.now();
  if (p > 0.6) state.hearingUntil = now + 300;
  const status = document.body.dataset.status;
  if (now < state.hearingUntil && (status === "listening" || status === "hearing")) setStatus("hearing");
  else if (status === "hearing" && !isSpeaking() && state.serverDone) setStatus("listening");
}

async function startVAD() {
  const noop = () => {};
  state.vad = await vad.MicVAD.new({
    model: "v5",
    baseAssetPath: `https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@${VAD_VERSION}/dist/`,
    onnxWASMBasePath: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
    onFrameProcessed: onFrame,
    onSpeechStart: noop, onSpeechRealStart: noop, onSpeechEnd: noop, onVADMisfire: noop,
  });
}

async function toggleCall() {
  const btn = $("call-btn");
  try {
    if (state.inCall) { endCall(); return; }
    await ensureAudio();
    if (!state.vad) {
      btn.disabled = true; btn.textContent = "Connecting…";
      await startVAD();
    }
    state.vad.start();
    state.inCall = true;
    sendConfig();
    send({ type: "call_start" });
    state.callStartedAt = Date.now();
    state.timer = setInterval(tickTimer, 1000);
    tickTimer();
    document.body.classList.add("in-call");
    btn.textContent = "End call";
    setStatus("listening");
  } catch (err) {
    console.error(err);
    addMessage("error", `Microphone or voice detection failed to start: ${err.message || err}`);
    btn.textContent = "Start call";
  } finally {
    btn.disabled = false;
  }
}

function endCall() {
  state.vad?.pause();
  state.inCall = false;
  send({ type: "call_end" });
  stopPlayback();
  dropLiveUser();
  clearInterval(state.timer);
  document.body.classList.remove("in-call");
  document.body.style.setProperty("--level", 0);
  $("call-btn").textContent = "Start call";
  setStatus("idle");
}

function tickTimer() {
  const s = Math.floor((Date.now() - state.callStartedAt) / 1000);
  $("call-timer").textContent = `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

// UI helpers
function langTag(lang) {
  const tag = document.createElement("span");
  tag.className = "lang";
  tag.textContent = lang === "hi" ? "Hindi" : "English";
  return tag;
}

function addMessage(role, text, lang) {
  $("log").querySelector(".empty")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  if (lang) el.appendChild(langTag(lang));
  const body = document.createElement("span");
  body.className = "text";
  body.textContent = text;
  el.appendChild(body);
  $("log").appendChild(el);
  scrollLog();
  return el;
}

function dropLiveUser() {
  state.liveUserEl?.remove();
  state.liveUserEl = null;
}

function showLatency(el, m) {
  const tag = document.createElement("span");
  tag.className = "latency";
  tag.textContent = `replied in ${(m.first_audio_ms / 1000).toFixed(2)} s`;
  const parts = [["STT", m.stt_ms], ["LLM first token", m.llm_first_ms], ["TTS first chunk", m.tts_first_ms]]
    .filter(([, v]) => v != null).map(([k, v]) => `${k} ${v} ms`);
  tag.title = `From the end of your speech to the first audio.\n${parts.join("\n")}`;
  el.appendChild(tag);
}

const scrollLog = () => { const log = $("log"); log.scrollTop = log.scrollHeight; };

async function sendText() {
  const input = $("text-input");
  const text = input.value.trim();
  if (!text) return;
  await ensureAudio();
  stopPlayback();
  send({ type: "text", text });
  input.value = "";
}

$("call-btn").addEventListener("click", toggleCall);
$("send-btn").addEventListener("click", sendText);
$("text-input").addEventListener("keydown", (e) => { if (e.key === "Enter") sendText(); });
$("language").addEventListener("change", sendConfig);
$("barge-in").addEventListener("change", sendConfig);
$("reset-btn").addEventListener("click", () => { stopPlayback(); send({ type: "reset" }); });

connect();
