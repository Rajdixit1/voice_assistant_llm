// Browser side of the voice assistant.
// - Silero VAD (vad-web) detects when the user starts/stops speaking.
// - Each finished utterance is sent to the server as 16 kHz float32 PCM.
// - The server streams back WAV chunks, which are queued for gapless playback.
// - Speaking while the assistant talks pauses playback; if it was real speech,
//   playback is cancelled and the new utterance starts a new turn (barge-in).

const VAD_VERSION = "0.0.24";
const ORT_VERSION = "1.22.0";
const $ = (id) => document.getElementById(id);

const state = {
  ws: null,
  vad: null,
  vadRunning: false,
  playCtx: null,
  currentTurn: 0,
  sources: new Set(),
  nextTime: 0,
  playChain: Promise.resolve(),
  serverDone: true,
  bargePending: false,
  ignoreUtterance: false,
  assistantEl: null,
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
const idleStatus = () => (state.vadRunning ? "listening" : "idle");

// WebSocket
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => { setStatus(idleStatus()); sendConfig(); };
  ws.onclose = () => { setStatus("offline"); setTimeout(connect, 1500); };
  ws.onmessage = (ev) => (typeof ev.data === "string" ? onMessage(JSON.parse(ev.data)) : onAudio(ev.data));
  state.ws = ws;
}
const send = (obj) => state.ws?.readyState === 1 && state.ws.send(JSON.stringify(obj));
const sendConfig = () => send({ type: "config", language: $("language").value });

function onMessage(msg) {
  switch (msg.type) {
    case "ready": {
      const voices = Object.entries(msg.tts).map(([l, v]) => `${l}: ${v}`).join(", ");
      $("models").textContent = `STT ${msg.stt} | LLM ${msg.llm} | TTS ${voices}`;
      break;
    }
    case "turn_start":
      state.currentTurn = msg.turn;
      state.serverDone = false;
      state.assistantEl = null;
      setStatus("thinking");
      break;
    case "transcript":
      if (msg.turn !== state.currentTurn) return;
      addMessage("user", msg.text, msg.lang);
      state.assistantEl = addMessage("assistant", "");
      break;
    case "token":
      if (msg.turn !== state.currentTurn || !state.assistantEl) return;
      state.assistantEl.querySelector(".text").textContent += msg.text;
      scrollLog();
      break;
    case "done":
      if (msg.turn === state.currentTurn) {
        state.serverDone = true;
        if (!isSpeaking()) setStatus(idleStatus());
      }
      break;
    case "no_speech":
      if (msg.turn === state.currentTurn) { state.serverDone = true; setStatus(idleStatus()); }
      break;
    case "reset_ok":
      $("log").innerHTML = "";
      break;
    case "error":
      addMessage("error", `Error: ${msg.message}`);
      state.serverDone = true;
      setStatus(idleStatus());
      break;
  }
}

// playback
function onAudio(buf) {
  const turn = new DataView(buf).getUint32(0, true);
  if (turn !== state.currentTurn || !state.playCtx) return;
  const wav = buf.slice(4);
  state.playChain = state.playChain.then(async () => {
    if (turn !== state.currentTurn) return;
    const audio = await state.playCtx.decodeAudioData(wav);
    if (turn !== state.currentTurn) return;
    const ctx = state.playCtx;
    const src = ctx.createBufferSource();
    src.buffer = audio;
    src.connect(ctx.destination);
    const start = Math.max(ctx.currentTime + 0.02, state.nextTime);
    src.start(start);
    state.nextTime = start + audio.duration;
    state.sources.add(src);
    setStatus("speaking");
    src.onended = () => {
      state.sources.delete(src);
      if (!state.sources.size && state.serverDone) setStatus(idleStatus());
    };
  }).catch((err) => console.error("playback", err));
}

function stopPlayback() {
  const wasActive = isSpeaking() || !state.serverDone;
  for (const src of state.sources) { src.onended = null; try { src.stop(); } catch (_) {} }
  state.sources.clear();
  state.nextTime = 0;
  state.currentTurn = -1;
  if (wasActive) state.assistantEl?.classList.add("cut");
  state.serverDone = true;
  if (state.playCtx?.state === "suspended") state.playCtx.resume();
}

// VAD
async function startVAD() {
  state.vad = await vad.MicVAD.new({
    model: "v5",
    baseAssetPath: `https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@${VAD_VERSION}/dist/`,
    onnxWASMBasePath: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
    positiveSpeechThreshold: 0.6,
    negativeSpeechThreshold: 0.4,
    redemptionFrames: 15, redemptionMs: 480,
    minSpeechFrames: 5, minSpeechMs: 160,
    preSpeechPadFrames: 8, preSpeechPadMs: 256,

    onSpeechStart: () => {
      if (isSpeaking()) {
        if (!$("barge-in").checked) { state.ignoreUtterance = true; return; }
        state.playCtx.suspend();
        state.bargePending = true;
      }
      state.ignoreUtterance = false;
      setStatus("hearing");
    },
    onVADMisfire: () => {
      if (state.bargePending) { state.bargePending = false; state.playCtx.resume(); setStatus("speaking"); }
      else if (!isSpeaking() && state.serverDone) setStatus(idleStatus());
    },
    onSpeechEnd: (audio) => {
      if (state.ignoreUtterance) { state.ignoreUtterance = false; return; }
      if (state.bargePending || isSpeaking() || !state.serverDone) stopPlayback();
      state.bargePending = false;
      if (state.ws?.readyState === 1) state.ws.send(audio);
      setStatus("thinking");
    },
  });
}

async function toggleListening() {
  const btn = $("start-btn");
  try {
    if (!state.playCtx) state.playCtx = new AudioContext();
    if (state.playCtx.state === "suspended") await state.playCtx.resume();
    if (!state.vad) {
      btn.disabled = true; btn.textContent = "Loading voice detection…";
      await startVAD();
    }
    if (state.vadRunning) { state.vad.pause(); state.vadRunning = false; }
    else { state.vad.start(); state.vadRunning = true; }
    btn.textContent = state.vadRunning ? "Stop listening" : "Start talking";
    if (!isSpeaking()) setStatus(idleStatus());
  } catch (err) {
    console.error(err);
    addMessage("error", `Microphone or voice detection failed to start: ${err.message || err}`);
    btn.textContent = "Start talking";
  } finally {
    btn.disabled = false;
  }
}

// UI helpers
function addMessage(role, text, lang) {
  $("log").querySelector(".empty")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  if (lang) {
    const tag = document.createElement("span");
    tag.className = "lang";
    tag.textContent = lang === "hi" ? "Hindi" : "English";
    el.appendChild(tag);
  }
  const body = document.createElement("span");
  body.className = "text";
  body.textContent = text;
  el.appendChild(body);
  $("log").appendChild(el);
  scrollLog();
  return el;
}
const scrollLog = () => { const log = $("log"); log.scrollTop = log.scrollHeight; };

function sendText() {
  const input = $("text-input");
  const text = input.value.trim();
  if (!text) return;
  if (!state.playCtx) state.playCtx = new AudioContext();
  stopPlayback();
  send({ type: "text", text });
  input.value = "";
}

$("start-btn").addEventListener("click", toggleListening);
$("send-btn").addEventListener("click", sendText);
$("text-input").addEventListener("keydown", (e) => { if (e.key === "Enter") sendText(); });
$("language").addEventListener("change", sendConfig);
$("reset-btn").addEventListener("click", () => { stopPlayback(); send({ type: "reset" }); });

connect();
