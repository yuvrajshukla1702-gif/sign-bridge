/**
 * Sign Bridge — dual-mode controller
 * Mode 1: Learning Hub (dictionary, pose matcher, quiz)
 * Mode 2: Live Conversation Bridge
 */

import { extractFeatures, extractPrimaryHand63, FEATURE_DIM, smoothHand63 } from "./landmarks.js";
import { PredictionDebouncer, SequenceBuffer, SentenceBuffer } from "./buffer.js";
import { evaluateLocal, HoldTracker, loadRefs, hasRefs, loadStaticModel, hasStaticModel, PASS_PCT, HOLD_MS } from "./coach.js";

const WINDOW = 30;
const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
const NUMBERS = "0123456789".split("");
const TOKEN_KEY = "signbridge.authToken";
const USER_KEY = "signbridge.user";
const MASTERY_KEY = "signbridge.mastery";
const HOLD_CIRC = 2 * Math.PI * 52;

const state = {
  view: "home", // home | login | signup | app
  user: null,
  mode: "learn", // learn | bridge
  learnTab: "practice",
  cameraOn: false,
  camera: null,
  tracker: null,
  trackerType: null, // hands | holistic
  smoothHand: null,
  labels: [],
  modelLoaded: false,
  staticReady: false,
  tokens: [],
  predicting: false,
  lastPredictAt: 0,
  lastEvalAt: 0,
  lastUiAt: 0,
  fps: 0,
  frames: 0,
  fpsTs: performance.now(),
  signsIndex: [],
  target: "A",
  quizTarget: "A",
  quizPassed: 0,
  quizLocked: false,
  practiceLocked: false,
  dictCat: "all",
  lessonPhase: "guide", // guide | match | hold | done
  jointErrors: null,
  lastHandLandmarks: null,
  mastery: {},
  hold: new HoldTracker(HOLD_MS, PASS_PCT),
  stepCount: 3,
  phrases: {},
  sentences: [],
  sentenceId: null,
  sentenceIdx: 0,
  sentenceTokenIdx: 0,
  sentenceLocked: false,
};

const seqBuf = new SequenceBuffer(WINDOW, FEATURE_DIM);
const debouncer = new PredictionDebouncer({ threshold: 0.85, minConsecutive: 10 });
const sentBuf = new SentenceBuffer({ idleMs: 2000 });

const $ = (id) => document.getElementById(id);
const els = {
  modelBadge: $("modelBadge"),
  camBadge: $("camBadge"),
  btnCamera: $("btnCamera"),
  cameraSelect: $("cameraSelect"),
  btnListen: $("btnListen"),
  video: $("inputVideo"),
  canvas: $("outputCanvas"),
  bridgeCanvas: $("bridgeCanvas"),
  predText: $("predText"),
  predMeta: $("predMeta"),
  bridgePredText: $("bridgePredText"),
  bridgePredMeta: $("bridgePredMeta"),
  matchBar: $("matchBar"),
  matchPct: $("matchPct"),
  coachTip: $("coachTip"),
  targetImg: $("targetImg"),
  targetKey: $("targetKey"),
  targetHint: $("targetHint"),
  targetRemember: $("targetRemember"),
  howToSteps: $("howToSteps"),
  targetPicker: $("targetPicker"),
  quizTarget: $("quizTarget"),
  quizStatus: $("quizStatus"),
  quizScore: $("quizScore"),
  quizMatchBar: $("quizMatchBar"),
  quizMatchPct: $("quizMatchPct"),
  btnQuizNext: $("btnQuizNext"),
  btnNextSign: $("btnNextSign"),
  btnReplayGuide: $("btnReplayGuide"),
  dictSearch: $("dictSearch"),
  dictGrid: $("dictGrid"),
  modeLearn: $("modeLearn"),
  modeBridge: $("modeBridge"),
  tokenChips: $("tokenChips"),
  chatLog: $("chatLog"),
  voiceText: $("voiceText"),
  signHits: $("signHits"),
  practiceGrid: $("practiceGrid"),
  emergencyBanner: $("emergencyBanner"),
  emergencyText: $("emergencyText"),
  fpsLabel: $("fpsLabel"),
  btnFlush: $("btnFlush"),
  btnClearTokens: $("btnClearTokens"),
  holdRingArc: $("holdRingArc"),
  holdLabel: $("holdLabel"),
  holdHud: $("holdHud"),
  lessonPhase: $("lessonPhase"),
  lessonPhases: $("lessonPhases"),
  stepProgress: $("stepProgress"),
  masteryCount: $("masteryCount"),
  masterFlash: $("masterFlash"),
  masterFlashText: $("masterFlashText"),
  guideFrame: document.querySelector(".guide-frame"),
  sentencePicker: $("sentencePicker"),
  sentenceEnglish: $("sentenceEnglish"),
  sentenceHint: $("sentenceHint"),
  sentenceTokens: $("sentenceTokens"),
  sentenceImg: $("sentenceImg"),
  sentenceTokenLabel: $("sentenceTokenLabel"),
  sentenceRemember: $("sentenceRemember"),
  sentenceTokenMeta: $("sentenceTokenMeta"),
  sentenceSteps: $("sentenceSteps"),
  sentenceStepProgress: $("sentenceStepProgress"),
  sentenceMatchBar: $("sentenceMatchBar"),
  sentenceMatchPct: $("sentenceMatchPct"),
  sentenceCoach: $("sentenceCoach"),
  btnSentenceGotIt: $("btnSentenceGotIt"),
  btnSentenceNext: $("btnSentenceNext"),
  liveCueImg: $("liveCueImg"),
  liveCueTitle: $("liveCueTitle"),
  liveCueTip: $("liveCueTip"),
};

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  const token = localStorage.getItem(TOKEN_KEY);
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(path, { ...options, headers });
  if (!res.ok) {
    let msg = await res.text();
    try {
      const j = JSON.parse(msg);
      msg = j.detail || msg;
    } catch (_) {}
    throw new Error(typeof msg === "string" ? msg : "Request failed");
  }
  return res.json();
}

function showView(name) {
  state.view = name;
  const map = { home: "viewHome", login: "viewLogin", signup: "viewSignup", app: "viewApp" };
  Object.entries(map).forEach(([k, id]) => {
    const el = $(id);
    if (el) el.hidden = k !== name;
  });
  if (name !== "app" && state.cameraOn) stopCamera();
}

function setSession(token, user) {
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
  state.user = user;
  const g = $("userGreeting");
  if (g) g.textContent = `Hi, ${user.name}`;
}

function clearSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  state.user = null;
}

async function restoreSession() {
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token) return false;
  if (token === "guest") {
    const raw = localStorage.getItem(USER_KEY);
    try {
      state.user = raw ? JSON.parse(raw) : { name: "Guest" };
    } catch {
      state.user = { name: "Guest" };
    }
    const g = $("userGreeting");
    if (g) g.textContent = `Hi, ${state.user.name || "Guest"}`;
    return true;
  }
  try {
    const data = await api("/api/auth/me");
    setSession(token, data.user);
    return true;
  } catch {
    clearSession();
    return false;
  }
}

function speak(text, { priority = false } = {}) {
  if (!text || !window.speechSynthesis) return;
  if (priority) window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = priority ? 1.05 : 1;
  window.speechSynthesis.speak(u);
}

function setMode(mode) {
  const prev = state.mode;
  state.mode = mode;
  document.querySelectorAll(".mode-btn").forEach((b) => {
    const on = b.dataset.mode === mode;
    b.classList.toggle("is-active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  });
  els.modeLearn.hidden = mode !== "learn";
  els.modeBridge.hidden = mode !== "bridge";
  els.modeLearn.classList.toggle("is-visible", mode === "learn");
  els.modeBridge.classList.toggle("is-visible", mode === "bridge");
  // Restart tracker when switching learn↔bridge (Hands vs Holistic)
  if (state.cameraOn && prev !== mode) {
    stopCamera();
    startCamera().catch((err) => console.warn(err));
  }
}

function setLearnTab(tab) {
  state.learnTab = tab;
  document.querySelectorAll("[data-learn-tab]").forEach((b) => {
    b.classList.toggle("is-active", b.dataset.learnTab === tab);
  });
  $("learnPractice").classList.toggle("is-visible", tab === "practice");
  const sent = $("learnSentences");
  if (sent) sent.classList.toggle("is-visible", tab === "sentences");
  $("learnQuiz").classList.toggle("is-visible", tab === "quiz");
  $("learnDict").classList.toggle("is-visible", tab === "dict");
  if (tab === "dict") loadDictionary();
  if (tab === "sentences") {
    state.hold.reset();
    state.sentenceLocked = false;
    renderSentencePicker();
    if (state.sentenceId) showSentenceToken();
    else if (state.sentences[0]) selectSentence(state.sentences[0].id);
  }
}

function dictUrl(key) {
  if (/^[A-Z]$/.test(key)) return `/assets/signs_dict/alphabet/${key}.png`;
  if (/^[0-9]$/.test(key)) return `/assets/signs_dict/numbers/${key}.png`;
  if (state.phrases?.[key]?.image) return state.phrases[key].image;
  return `/assets/signs/${key}.png`;
}

function updateLiveCue({ title, tip, src }) {
  if (els.liveCueTitle) els.liveCueTitle.textContent = title || "";
  if (els.liveCueTip) els.liveCueTip.textContent = tip || "Copy this handshape";
  if (els.liveCueImg && src) els.liveCueImg.src = src.includes("?") ? src : `${src}?v=${Date.now()}`;
}

function loadMastery() {
  try {
    state.mastery = JSON.parse(localStorage.getItem(MASTERY_KEY) || "{}") || {};
  } catch {
    state.mastery = {};
  }
}

function saveMastery() {
  localStorage.setItem(MASTERY_KEY, JSON.stringify(state.mastery));
}

function setLessonPhase(phase) {
  state.lessonPhase = phase;
  if (els.lessonPhase) {
    const labels = {
      guide: "Guide",
      match: "Matching…",
      hold: "Hold steady",
      done: "Mastered",
    };
    els.lessonPhase.textContent = labels[phase] || phase;
    els.lessonPhase.dataset.phase = phase;
  }
  if (els.lessonPhases) {
    els.lessonPhases.querySelectorAll("[data-phase]").forEach((el) => {
      const p = el.getAttribute("data-phase");
      el.classList.toggle("is-active", p === phase);
      el.classList.toggle("is-done", ["guide", "match", "hold", "done"].indexOf(p) < ["guide", "match", "hold", "done"].indexOf(phase));
    });
  }
  if (els.guideFrame) {
    els.guideFrame.classList.toggle("is-dim", phase !== "guide" && state.cameraOn);
    els.guideFrame.hidden = state.cameraOn && phase !== "guide" && phase !== "match";
  }
}

function updateHoldRing(progress) {
  if (!els.holdRingArc) return;
  const offset = HOLD_CIRC * (1 - progress);
  els.holdRingArc.style.strokeDasharray = `${HOLD_CIRC}`;
  els.holdRingArc.style.strokeDashoffset = `${offset}`;
  els.holdHud?.classList.toggle("is-holding", progress > 0.05 && progress < 1);
  els.holdHud?.classList.toggle("is-complete", progress >= 1);
  if (els.holdLabel) {
    if (progress >= 1) els.holdLabel.textContent = "Done";
    else if (progress > 0.05) els.holdLabel.textContent = "Hold";
    else els.holdLabel.textContent = "Match";
  }
}

function markMastered(key) {
  state.mastery[key] = (state.mastery[key] || 0) + 1;
  saveMastery();
  refreshMasteryUi();
  if (els.masterFlash && els.masterFlashText) {
    els.masterFlashText.textContent = `Mastered “${key}”!`;
    els.masterFlash.hidden = false;
    els.masterFlash.classList.remove("is-show");
    // reflow
    void els.masterFlash.offsetWidth;
    els.masterFlash.classList.add("is-show");
    setTimeout(() => {
      els.masterFlash.hidden = true;
      els.masterFlash.classList.remove("is-show");
    }, 1400);
  }
  speak(`Nice. ${key}`);
}

function refreshMasteryUi() {
  const n = Object.keys(state.mastery).length;
  if (els.masteryCount) els.masteryCount.textContent = `${n} mastered`;
  if (!els.targetPicker) return;
  els.targetPicker.querySelectorAll("[data-sign]").forEach((btn) => {
    const k = btn.getAttribute("data-sign");
    btn.classList.toggle("is-mastered", !!state.mastery[k]);
    btn.classList.toggle("is-current", k === state.target);
  });
}

function nextSignKey(current) {
  const pool = [...ALPHABET, ...NUMBERS];
  const i = pool.indexOf(current);
  return pool[(i + 1) % pool.length];
}

async function setTarget(key) {
  // Motion phrases (thank_you, water, …) aren't letter-scored — open Sentences coach
  if (state.phrases?.[key] || (!/^[A-Z0-9]$/.test(key) && key.length > 1)) {
    if (!state.phrases?.[key] && state.sentences.length) {
      // still try sentences after load
    }
    const phraseSentence =
      state.sentences.find((s) => s.id === key) ||
      state.sentences.find((s) => s.tokens?.length === 1 && s.tokens[0].key === key);
    setLearnTab("sentences");
    if (phraseSentence) {
      selectSentence(phraseSentence.id);
      return;
    }
    if (state.phrases?.[key]) {
      // Build a one-off practice view via synthetic selection
      state.sentenceId = key;
      if (!state.sentences.some((s) => s.id === key)) {
        state.sentences = [
          {
            id: key,
            english: (state.phrases[key].title || key).replaceAll("_", " "),
            hint: state.phrases[key].remember || "",
            tokens: [{ kind: "phrase", key, label: state.phrases[key].title || key }],
          },
          ...state.sentences,
        ];
      }
      selectSentence(key);
      return;
    }
  }

  state.target = key;
  state.hold.reset();
  state.practiceLocked = false;
  updateHoldRing(0);
  setLessonPhase("guide");
  els.targetKey.textContent = key;
  els.targetImg.src = dictUrl(key) + `?v=${Date.now()}`;
  els.targetHint.textContent = "Read the checklist, start camera, then shape your hand until steps light up.";
  els.matchBar.style.width = "0%";
  els.matchPct.textContent = "0%";
  els.predText.textContent = "—";
  els.coachTip.textContent = "Start with the checklist — then match live in the camera.";
  refreshMasteryUi();
  updateLiveCue({
    title: key,
    tip: "Match this letter / number in the camera",
    src: dictUrl(key),
  });

  try {
    const info = await api(`/api/instructions/${encodeURIComponent(key)}`);
    els.targetRemember.textContent = info.remember || "";
    const steps = info.steps || [];
    state.stepCount = Math.max(1, steps.length);
    els.howToSteps.innerHTML = steps
      .map(
        (s, i) =>
          `<li data-step="${i}" class="step-item"><span class="step-check" aria-hidden="true"></span><span class="step-text">${escapeHtml(s)}</span></li>`
      )
      .join("");
    if (els.stepProgress) els.stepProgress.textContent = `0 / ${state.stepCount}`;
    if (info.image_url) {
      els.targetImg.src = info.image_url + `?v=${Date.now()}`;
      updateLiveCue({ title: key, tip: info.remember || "", src: info.image_url });
    } else {
      updateLiveCue({ title: key, tip: info.remember || "", src: dictUrl(key) });
    }
  } catch {
    els.targetRemember.textContent = `Practice sign “${key}”`;
    els.howToSteps.innerHTML = `<li data-step="0" class="step-item"><span class="step-check"></span><span class="step-text">Look up how to form “${escapeHtml(key)}”.</span></li>`;
    state.stepCount = 1;
  }
}

function buildTargetPicker() {
  els.targetPicker.innerHTML = "";
  [...ALPHABET, ...NUMBERS].forEach((k) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-practice";
    btn.dataset.sign = k;
    btn.textContent = k;
    btn.title = state.mastery[k] ? `Mastered ×${state.mastery[k]}` : `Practice ${k}`;
    btn.addEventListener("click", () => setTarget(k));
    els.targetPicker.appendChild(btn);
  });
  refreshMasteryUi();
}

function applyStepUnlock(unlock) {
  const items = els.howToSteps?.querySelectorAll(".step-item") || [];
  items.forEach((li, i) => {
    li.classList.toggle("is-done", i < unlock);
    li.classList.toggle("is-current", i === unlock && unlock < state.stepCount);
  });
  if (els.stepProgress) {
    els.stepProgress.textContent = `${Math.min(unlock, state.stepCount)} / ${state.stepCount}`;
  }
}

function randomPrompt() {
  const pool = [...ALPHABET, ...NUMBERS];
  return pool[Math.floor(Math.random() * pool.length)];
}

function nextQuiz() {
  state.quizTarget = randomPrompt();
  state.quizLocked = false;
  state.hold.reset();
  updateHoldRing(0);
  els.quizTarget.textContent = state.quizTarget;
  els.quizStatus.textContent = "Form the sign, then hold…";
  if (els.quizMatchBar) els.quizMatchBar.style.width = "0%";
  if (els.quizMatchPct) els.quizMatchPct.textContent = "0%";
}

function renderTokens() {
  if (!state.tokens.length) {
    els.tokenChips.innerHTML = `<span class="chips-empty">Detected signs appear here</span>`;
    return;
  }
  els.tokenChips.innerHTML = state.tokens.map((t) => `<span class="chip">${escapeHtml(t)}</span>`).join("");
}

function addChat(role, text, kind = "") {
  const item = document.createElement("div");
  item.className = `chat-item ${kind}`.trim();
  item.innerHTML = `<span class="role">${escapeHtml(role)}</span>${escapeHtml(text)}`;
  els.chatLog.appendChild(item);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

async function polishAndSpeak(tokens) {
  if (!tokens?.length) return;
  try {
    const data = await api("/api/polish", {
      method: "POST",
      body: JSON.stringify({ tokens, backend: "rule" }),
    });
    addChat("sign → speech", data.sentence, "sign");
    speak(data.sentence);
  } catch {
    const fallback = tokens.join(" ") + ".";
    addChat("sign → speech", fallback, "sign");
    speak(fallback);
  }
}

function onToken(token) {
  const eos = sentBuf.add(token);
  state.tokens = [...sentBuf.tokens];
  renderTokens();
  if (eos) {
    state.tokens = [];
    renderTokens();
    polishAndSpeak(eos);
  }
}

setInterval(() => {
  const flushed = sentBuf.checkIdle();
  if (flushed) {
    state.tokens = [];
    renderTokens();
    polishAndSpeak(flushed);
  }
}, 400);

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    state.modelLoaded = h.model_loaded;
    state.staticReady = h.static_ready;
    state.labels = h.labels || [];
    els.modelBadge.textContent = h.model_loaded
      ? "LSTM ready"
      : h.static_ready
        ? "Learn mode ready"
        : "Practice mode";
    els.modelBadge.className = `badge ${h.model_loaded || h.static_ready ? "badge-ok" : "badge-warn"}`;
    buildPhraseInject(state.labels.length ? state.labels : ["hello", "thank_you", "water", "need_help", "EOS"]);
  } catch {
    els.modelBadge.textContent = "API offline";
    els.modelBadge.className = "badge badge-warn";
  }
}

function buildPhraseInject(labels) {
  els.practiceGrid.innerHTML = "";
  labels.forEach((label) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-practice";
    btn.textContent = label.replaceAll("_", " ");
    btn.addEventListener("click", () => {
      els.bridgePredText.textContent = label;
      els.bridgePredMeta.textContent = "Injected token";
      onToken(label);
    });
    els.practiceGrid.appendChild(btn);
  });
}

async function loadSignsIndex() {
  try {
    const data = await api("/api/signs");
    state.signsIndex = data.signs || [];
  } catch {
    state.signsIndex = [];
  }
}

async function loadDictionary() {
  const q = els.dictSearch.value.trim();
  const cat = state.dictCat;
  const data = await api(`/api/dictionary?q=${encodeURIComponent(q)}&category=${encodeURIComponent(cat)}`);
  els.dictGrid.innerHTML = (data.items || [])
    .map(
      (it) => `
      <button type="button" class="dict-card" data-key="${escapeHtml(it.key)}" data-cat="${escapeHtml(it.category)}">
        <img src="${it.url}" alt="${escapeHtml(it.label)}" />
        <span>${escapeHtml(it.label)}</span>
      </button>`
    )
    .join("");
  els.dictGrid.querySelectorAll(".dict-card").forEach((card) => {
    card.addEventListener("click", () => {
      const key = card.getAttribute("data-key");
      const cat = card.getAttribute("data-cat");
      if (cat === "phrases" || (state.phrases && state.phrases[key])) {
        setLearnTab("sentences");
        const sid =
          state.sentences.find((s) => s.id === key)?.id ||
          state.sentences.find((s) => s.tokens?.some((t) => t.key === key))?.id;
        if (sid) selectSentence(sid);
        else setTarget(key);
        return;
      }
      setTarget(key);
      setLearnTab("practice");
    });
  });
}

function matchSigns(text) {
  const t = text.toLowerCase();
  const keys = [...state.signsIndex].sort((a, b) => b.key.length - a.key.length);
  const hits = [];
  let remaining = t;
  for (const item of keys) {
    const pattern = item.key.replaceAll("_", "[\\s_]+");
    const re = new RegExp(pattern, "i");
    if (re.test(remaining)) {
      hits.push(item);
      remaining = remaining.replace(re, " ");
    }
  }
  return hits;
}

function showSignHits(hits) {
  if (!hits.length) {
    els.signHits.innerHTML = `<p class="hint">No matching asset. Add GIFs under assets/signs/.</p>`;
    return;
  }
  els.signHits.innerHTML = hits
    .map(
      (h) => `<figure><img src="${h.url}" alt="${escapeHtml(h.key)}" /><figcaption>${escapeHtml(
        h.key.replaceAll("_", " ")
      )}</figcaption></figure>`
    )
    .join("");
}

function listenVoice() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    els.voiceText.textContent = "Speech recognition needs Chrome/Edge.";
    return;
  }
  const rec = new SR();
  rec.lang = "en-US";
  rec.interimResults = false;
  els.voiceText.textContent = "Listening…";
  els.btnListen.disabled = true;
  rec.onresult = (ev) => {
    const text = ev.results[0][0].transcript;
    els.voiceText.textContent = `Heard: “${text}”`;
    addChat("voice → sign", text, "voice");
    showSignHits(matchSigns(text));
  };
  rec.onerror = () => {
    els.voiceText.textContent = "Could not hear clearly — try again.";
  };
  rec.onend = () => {
    els.btnListen.disabled = false;
  };
  rec.start();
}

async function triggerEmergency(key) {
  const data = await api("/api/emergency", {
    method: "POST",
    body: JSON.stringify({ key }),
  });
  els.emergencyBanner.hidden = false;
  els.emergencyText.textContent = data.label;
  addChat("emergency", data.spoken, "emergency");
  speak(data.spoken, { priority: true });
  onToken(data.gloss);
  setTimeout(() => {
    els.emergencyBanner.hidden = true;
  }, 4000);
}

function currentSentence() {
  return state.sentences.find((s) => s.id === state.sentenceId) || null;
}

function currentSentenceToken() {
  const s = currentSentence();
  if (!s) return null;
  return s.tokens[state.sentenceTokenIdx] || null;
}

function renderSentencePicker() {
  if (!els.sentencePicker) return;
  els.sentencePicker.innerHTML = state.sentences
    .map((s) => {
      const active = s.id === state.sentenceId ? "is-active" : "";
      const mastered = state.mastery[`sent:${s.id}`] ? "is-mastered" : "";
      return `<button type="button" class="sentence-chip ${active} ${mastered}" data-sentence="${escapeHtml(s.id)}">
        <span class="sentence-chip-en">${escapeHtml(s.english)}</span>
        <span class="sentence-chip-lvl">${escapeHtml(s.level || "")}</span>
      </button>`;
    })
    .join("");
  els.sentencePicker.querySelectorAll("[data-sentence]").forEach((btn) => {
    btn.addEventListener("click", () => selectSentence(btn.getAttribute("data-sentence")));
  });
}

function selectSentence(id) {
  state.sentenceId = id;
  state.sentenceTokenIdx = 0;
  state.sentenceLocked = false;
  state.hold.reset();
  updateHoldRing(0);
  renderSentencePicker();
  showSentenceToken();
}

function showSentenceToken() {
  const sent = currentSentence();
  const tok = currentSentenceToken();
  if (!sent || !tok) return;

  state.hold.reset();
  state.sentenceLocked = false;
  updateHoldRing(0);
  setLessonPhase("guide");

  if (els.sentenceEnglish) els.sentenceEnglish.textContent = sent.english;
  if (els.sentenceHint) els.sentenceHint.textContent = sent.hint || "";

  if (els.sentenceTokens) {
    els.sentenceTokens.innerHTML = sent.tokens
      .map((t, i) => {
        const cls =
          i < state.sentenceTokenIdx ? "is-done" : i === state.sentenceTokenIdx ? "is-current" : "";
        return `<span class="sent-token ${cls}">${escapeHtml(t.label || t.key)}</span>`;
      })
      .join('<span class="sent-arrow">→</span>');
  }

  const phrase = tok.kind === "phrase" ? state.phrases[tok.key] : null;
  const label = tok.label || tok.key;
  if (els.sentenceTokenLabel) els.sentenceTokenLabel.textContent = label;
  if (els.sentenceRemember) {
    els.sentenceRemember.textContent =
      phrase?.remember ||
      (tok.kind === "letter" ? `Hold letter “${tok.key}” clearly` : `Sign “${label}”`);
  }
  if (els.sentenceTokenMeta) {
    els.sentenceTokenMeta.textContent = `Part ${state.sentenceTokenIdx + 1} of ${sent.tokens.length} · ${tok.kind}`;
  }
  if (els.sentenceStepProgress) {
    els.sentenceStepProgress.textContent = `${state.sentenceTokenIdx + 1} / ${sent.tokens.length}`;
  }

  const img =
    phrase?.image ||
    (tok.kind === "letter" ? dictUrl(tok.key) : `/assets/signs/${tok.key}.png`);
  if (els.sentenceImg) {
    els.sentenceImg.src = img + `?v=1`;
    els.sentenceImg.alt = label;
  }
  updateLiveCue({
    title: label,
    tip: phrase?.remember || (tok.kind === "letter" ? `Hold letter “${tok.key}”` : `Copy “${label}”`),
    src: img,
  });

  const steps =
    phrase?.steps ||
    (tok.kind === "letter"
      ? [
          `Form the letter “${tok.key}” the same way as in Letters mode.`,
          "Hold it steady in the guide zone until the ring fills.",
          "Then we’ll move to the next part of the sentence.",
        ]
      : [`Perform the “${label}” sign, then hold still or tap “I signed it”.`]);

  if (els.sentenceSteps) {
    els.sentenceSteps.innerHTML = steps
      .map(
        (s, i) =>
          `<li data-step="${i}" class="step-item"><span class="step-check"></span><span class="step-text">${escapeHtml(s)}</span></li>`
      )
      .join("");
  }

  if (els.sentenceMatchBar) els.sentenceMatchBar.style.width = "0%";
  if (els.sentenceMatchPct) els.sentenceMatchPct.textContent = "0%";
  if (els.sentenceCoach) {
    els.sentenceCoach.textContent =
      tok.kind === "letter"
        ? `Match letter “${tok.key}” live, then hold.`
        : `Do the motion for “${label}”, keep your hand in frame, then hold — or tap “I signed it”.`;
  }

  // Mirror into main overlay for consistency while on sentences tab
  if (els.predMeta) els.predMeta.textContent = `Sentence: ${sent.english} · now “${label}”`;
}

function advanceSentenceToken() {
  const sent = currentSentence();
  if (!sent || state.sentenceLocked) return;
  state.sentenceLocked = true;
  const tok = currentSentenceToken();
  if (tok) markMastered(tok.kind === "letter" ? tok.key : `phrase:${tok.key}`);

  if (state.sentenceTokenIdx < sent.tokens.length - 1) {
    state.sentenceTokenIdx += 1;
    speak(sent.tokens[state.sentenceTokenIdx].label || sent.tokens[state.sentenceTokenIdx].key);
    setTimeout(() => {
      state.sentenceLocked = false;
      showSentenceToken();
    }, 550);
  } else {
    setLessonPhase("done");
    markMastered(`sent:${sent.id}`);
    if (els.sentenceCoach) els.sentenceCoach.textContent = `Sentence complete: “${sent.english}”`;
    if (els.masterFlashText) els.masterFlashText.textContent = `Sentence done!`;
    if (els.masterFlash) {
      els.masterFlash.hidden = false;
      els.masterFlash.classList.add("is-show");
      setTimeout(() => {
        els.masterFlash.hidden = true;
        els.masterFlash.classList.remove("is-show");
      }, 1400);
    }
    speak(sent.english);
    renderSentencePicker();
    setTimeout(() => {
      const i = state.sentences.findIndex((s) => s.id === sent.id);
      const next = state.sentences[(i + 1) % state.sentences.length];
      if (next) selectSentence(next.id);
    }, 1600);
  }
}

async function loadSentences() {
  try {
    const data = await api("/api/sentences");
    state.phrases = data.phrases || {};
    state.sentences = data.sentences || [];
    if (state.sentences[0]) selectSentence(state.sentences[0].id);
    else renderSentencePicker();
  } catch (err) {
    console.warn("sentences load failed", err);
    state.sentences = [];
  }
}

function evaluateHand(hand63) {
  const now = performance.now();

  if (state.learnTab === "sentences") {
    evaluateSentenceHand(hand63, now);
    return;
  }

  const target = state.learnTab === "quiz" ? state.quizTarget : state.target;
  if (!hasRefs() && !hasStaticModel()) return;

  const data = evaluateLocal(target, hand63);
  const pct = data.match_pct || 0;
  state.jointErrors = data.jointErrors;

  const hold = state.hold.update(pct, now);
  updateHoldRing(hold.progress);

  if (now - state.lastUiAt < 40) return;
  state.lastUiAt = now;

  els.matchBar.style.width = `${pct}%`;
  els.matchBar.classList.toggle("is-good", pct >= PASS_PCT);
  els.matchPct.textContent = `${pct.toFixed(0)}%`;
  els.predText.textContent = `${pct.toFixed(0)}%`;
  els.coachTip.textContent = data.tip || "";
  applyStepUnlock(data.stepUnlock || 0);

  if (els.quizMatchBar) {
    els.quizMatchBar.style.width = `${pct}%`;
    els.quizMatchBar.classList.toggle("is-good", pct >= PASS_PCT);
  }
  if (els.quizMatchPct) els.quizMatchPct.textContent = `${pct.toFixed(0)}%`;

  if (pct < 28) {
    if (state.lessonPhase !== "done") setLessonPhase(state.cameraOn ? "match" : "guide");
    els.predMeta.textContent =
      data.scorer === "trained_mlp" && data.tip ? data.tip : `Target ${target} · shape your hand`;
  } else if (!hold.holding && !hold.done) {
    setLessonPhase("match");
    els.predMeta.textContent = data.tip || `Target ${target}`;
  } else if (hold.holding && !hold.done) {
    setLessonPhase("hold");
    els.predMeta.textContent = `Hold still… ${Math.round(hold.progress * 100)}%`;
  }

  if (hold.done) {
    setLessonPhase("done");
    els.predMeta.textContent = `Mastered “${target}”`;
    if (state.learnTab === "quiz" && !state.quizLocked) {
      state.quizLocked = true;
      state.quizPassed += 1;
      els.quizScore.textContent = String(state.quizPassed);
      els.quizStatus.textContent = `Cleared “${target}”!`;
      markMastered(target);
      setTimeout(nextQuiz, 850);
    } else if (state.learnTab === "practice" && !state.practiceLocked) {
      state.practiceLocked = true;
      markMastered(target);
    }
  }
}

function evaluateSentenceHand(hand63, now) {
  const tok = currentSentenceToken();
  if (!tok || state.sentenceLocked) return;
  if (!hasRefs() && !hasStaticModel()) return;

  const data = evaluateLocal(tok.key, hand63);
  const pct = data.match_pct || 0;
  state.jointErrors = data.jointErrors;
  const hold = state.hold.update(pct, now);
  updateHoldRing(hold.progress);

  if (now - state.lastUiAt < 40) return;
  state.lastUiAt = now;

  if (els.sentenceMatchBar) {
    els.sentenceMatchBar.style.width = `${pct}%`;
    els.sentenceMatchBar.classList.toggle("is-good", pct >= PASS_PCT);
  }
  if (els.sentenceMatchPct) els.sentenceMatchPct.textContent = `${pct.toFixed(0)}%`;
  if (els.predText) els.predText.textContent = `${pct.toFixed(0)}%`;
  if (els.sentenceCoach) els.sentenceCoach.textContent = data.tip || "";
  applySentenceSteps(data.stepUnlock || 0);

  if (hold.holding) setLessonPhase("hold");
  else if (pct >= 28) setLessonPhase("match");

  if (hold.done) advanceSentenceToken();
}

function applySentenceSteps(unlock) {
  const items = els.sentenceSteps?.querySelectorAll(".step-item") || [];
  items.forEach((li, i) => {
    li.classList.toggle("is-done", i < unlock);
    li.classList.toggle("is-current", i === unlock);
  });
}

async function predictWindow(window) {
  if (state.predicting || state.mode !== "bridge") return;
  const now = performance.now();
  if (now - state.lastPredictAt < 120) return;
  state.lastPredictAt = now;
  state.predicting = true;
  try {
    const data = await api("/api/predict", {
      method: "POST",
      body: JSON.stringify({ window }),
    });
    if (!data.model_loaded) {
      els.bridgePredText.textContent = "—";
      els.bridgePredMeta.textContent = data.message || "Train LSTM for live phrase recognition";
      return;
    }
    els.bridgePredText.textContent = data.label;
    els.bridgePredMeta.textContent = `confidence ${data.confidence.toFixed(2)}`;
    const fired = debouncer.update(data.label, data.confidence);
    if (fired) onToken(fired);
  } finally {
    state.predicting = false;
  }
}

function jointColor(err) {
  if (err == null) return "#5ed1cb";
  if (err < 0.06) return "#5ed1cb";
  if (err < 0.12) return "#f0c45c";
  return "#e85d4c";
}

function drawResults(canvas, results) {
  const ctx = canvas.getContext("2d", { alpha: false });
  const video = els.video;
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.save();
  ctx.clearRect(0, 0, w, h);
  ctx.translate(w, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(results.image || video, 0, 0, w, h);

  if (window.drawConnectors && window.drawLandmarks) {
    const handConn = window.HAND_CONNECTIONS;
    const poseConn = window.POSE_CONNECTIONS;
    const hands = [];
    if (results.multiHandLandmarks?.length) {
      results.multiHandLandmarks.forEach((h) => hands.push(h));
    } else {
      if (results.leftHandLandmarks) hands.push(results.leftHandLandmarks);
      if (results.rightHandLandmarks) hands.push(results.rightHandLandmarks);
    }
    for (const hand of hands) {
      if (!hand || !handConn) continue;
      const errs = state.mode === "learn" ? state.jointErrors : null;
      const avgErr = errs ? [...errs].reduce((a, b) => a + b, 0) / errs.length : 0.05;
      const bone = jointColor(avgErr);
      drawConnectors(ctx, hand, handConn, { color: bone, lineWidth: 3 });
      // Per-joint colors for coaching
      for (let i = 0; i < hand.length; i++) {
        const lm = hand[i];
        const c = jointColor(errs ? errs[i] : 0.04);
        ctx.beginPath();
        ctx.fillStyle = c;
        ctx.arc(lm.x * w, lm.y * h, i % 4 === 0 ? 5 : 3.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (results.poseLandmarks && poseConn && state.mode === "bridge") {
      drawConnectors(ctx, results.poseLandmarks, poseConn, { color: "#5ed1cb44", lineWidth: 2 });
    }
  }
  ctx.restore();
}

function onResults(results) {
  const canvas = state.mode === "bridge" ? els.bridgeCanvas : els.canvas;
  if (state.mode === "learn" && results.multiHandLandmarks?.[0]) {
    state.lastHandLandmarks = results.multiHandLandmarks[0];
  }
  drawResults(canvas, results);

  if (state.mode === "bridge") {
    const features = extractFeatures(results);
    seqBuf.push(features);
    if (seqBuf.ready() && state.modelLoaded) predictWindow(seqBuf.getWindow());
  }

  const aspect = (els.video.videoWidth || 640) / (els.video.videoHeight || 480);
  let hand = extractPrimaryHand63(results, aspect);
  if (hand) {
    state.smoothHand = smoothHand63(state.smoothHand, hand, 0.55);
    hand = state.smoothHand;
  }
  if (state.mode === "learn") {
    if (hand) evaluateHand(hand);
    else {
      state.jointErrors = null;
      if (performance.now() - state.lastUiAt > 200) {
        state.lastUiAt = performance.now();
        els.coachTip.textContent = "Bring one hand into the glowing guide zone.";
        els.predMeta.textContent = "Waiting for hand…";
        if (state.lessonPhase !== "done") setLessonPhase("guide");
        updateHoldRing(0);
        state.hold.reset();
      }
    }
  }

  state.frames += 1;
  const t = performance.now();
  if (t - state.fpsTs >= 1000) {
    state.fps = state.frames;
    state.frames = 0;
    state.fpsTs = t;
    els.fpsLabel.textContent = `${state.fps} FPS`;
  }
}

const CAMERA_STORE_KEY = "signbridge.cameraDeviceId";

const PHONE_CAMERA_RE =
  /iphone|ipad|android| continuity|phone link|phone camera| continuity camera|continuitycam|epoccam|iriun|droidcam|ivcam|camo|nds-?webcam|obs virtual|virtual camera|manycam|snap camera/i;

const LAPTOP_CAMERA_RE =
  /integrated|built[- ]?in|laptop|face.?time|hd webcam|usb.?2\.0.?camera|usb camera|realtek|logitech c2|microsoft.?life|front camera|user facing/i;

function scoreCameraLabel(label = "") {
  const l = label.toLowerCase();
  if (PHONE_CAMERA_RE.test(l)) return -100;
  if (LAPTOP_CAMERA_RE.test(l)) return 50;
  if (/webcam|camera|hd|usb/.test(l)) return 20;
  return 0;
}

function pickPreferredCamera(devices) {
  const saved = localStorage.getItem(CAMERA_STORE_KEY);
  if (saved && devices.some((d) => d.deviceId === saved)) {
    return saved;
  }
  const ranked = [...devices].sort((a, b) => scoreCameraLabel(b.label) - scoreCameraLabel(a.label));
  // Prefer non-phone cameras first
  const nonPhone = ranked.filter((d) => scoreCameraLabel(d.label) >= 0);
  return (nonPhone[0] || ranked[0])?.deviceId || "";
}

async function listVideoDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const all = await navigator.mediaDevices.enumerateDevices();
  return all.filter((d) => d.kind === "videoinput");
}

function fillCameraSelect(devices, selectedId) {
  const sel = els.cameraSelect;
  if (!sel) return;
  if (!devices.length) {
    sel.innerHTML = `<option value="">No cameras found</option>`;
    return;
  }
  sel.innerHTML = devices
    .map((d) => {
      const label = d.label || `Camera ${d.deviceId.slice(0, 6)}`;
      const tag = scoreCameraLabel(label) < 0 ? " (phone/virtual)" : "";
      const selected = d.deviceId === selectedId ? "selected" : "";
      return `<option value="${escapeHtml(d.deviceId)}" ${selected}>${escapeHtml(label)}${tag}</option>`;
    })
    .join("");
}

async function refreshCameraList(requestPermission = false) {
  try {
    if (requestPermission) {
      // Labels are blank until permission is granted once
      const tmp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      tmp.getTracks().forEach((t) => t.stop());
    }
    const devices = await listVideoDevices();
    const preferred = pickPreferredCamera(devices);
    fillCameraSelect(devices, preferred);
    if (preferred) localStorage.setItem(CAMERA_STORE_KEY, preferred);
    return devices;
  } catch (err) {
    console.warn("camera list failed", err);
    if (els.cameraSelect) {
      els.cameraSelect.innerHTML = `<option value="">Allow camera to list devices</option>`;
    }
    return [];
  }
}

function selectedCameraId() {
  return els.cameraSelect?.value || localStorage.getItem(CAMERA_STORE_KEY) || "";
}

async function startCamera() {
  if (state.cameraOn) {
    stopCamera();
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    els.predMeta.textContent = "Camera API missing in this window. Open http://127.0.0.1:8000 in Chrome/Edge.";
    els.camBadge.textContent = "No camera API";
    els.camBadge.className = "badge badge-warn";
    return;
  }

  els.predMeta.textContent = "Selecting laptop camera…";
  els.camBadge.textContent = "Asking permission…";

  // Refresh device list (with permission) so we can avoid phone cameras
  await refreshCameraList(true);
  let deviceId = selectedCameraId();
  const devices = await listVideoDevices();
  if (!deviceId) deviceId = pickPreferredCamera(devices);

  const chosen = devices.find((d) => d.deviceId === deviceId);
  if (chosen && scoreCameraLabel(chosen.label) < 0) {
    // User still has a phone camera selected — warn and try a better one
    const better = pickPreferredCamera(devices.filter((d) => scoreCameraLabel(d.label) >= 0));
    if (better) {
      deviceId = better;
      fillCameraSelect(devices, deviceId);
      localStorage.setItem(CAMERA_STORE_KEY, deviceId);
    }
  }

  const videoConstraints = state.mode === "learn"
    ? { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } }
    : { width: { ideal: 960 }, height: { ideal: 540 }, frameRate: { ideal: 24, max: 30 } };
  if (deviceId) {
    videoConstraints.deviceId = { exact: deviceId };
  } else {
    videoConstraints.facingMode = "user";
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: videoConstraints,
    });
  } catch (err) {
    // Fallback without exact deviceId
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
    } catch (err2) {
      const name = err2?.name || err?.name || "Error";
      const msg = err2?.message || err?.message || String(err2 || err);
      console.error("getUserMedia failed", err2 || err);
      els.camBadge.textContent = "Camera blocked";
      els.camBadge.className = "badge badge-warn";
      if (name === "NotAllowedError" || name === "PermissionDeniedError") {
        els.predMeta.textContent =
          "Camera permission denied. Allow camera, then pick your laptop webcam from the Camera list.";
      } else if (name === "NotFoundError" || name === "DevicesNotFoundError") {
        els.predMeta.textContent = "No webcam found. Enable the laptop camera in Windows settings.";
      } else if (name === "NotReadableError" || name === "TrackStartError") {
        els.predMeta.textContent = "Camera is busy (Zoom/Teams/Phone Link?). Close other apps.";
      } else {
        els.predMeta.textContent = `Camera failed: ${name} — ${msg}`;
      }
      return;
    }
  }

  // Show which camera is active
  const track = stream.getVideoTracks()[0];
  const settings = track?.getSettings?.() || {};
  const activeLabel =
    devices.find((d) => d.deviceId === (settings.deviceId || deviceId))?.label ||
    track?.label ||
    "Camera";
  if (settings.deviceId) {
    localStorage.setItem(CAMERA_STORE_KEY, settings.deviceId);
    fillCameraSelect(devices, settings.deviceId);
  }

  els.video.srcObject = stream;
  els.video.muted = true;
  els.video.playsInline = true;
  await els.video.play();

  els.predMeta.textContent = `Using: ${activeLabel}`;
  if (scoreCameraLabel(activeLabel) < 0) {
    els.predMeta.textContent =
      `Using phone/virtual camera (“${activeLabel}”). Pick your laptop webcam in the Camera dropdown, then Start again.`;
    els.camBadge.textContent = "Wrong camera";
    els.camBadge.className = "badge badge-warn";
  }

  const useHands = state.mode === "learn";
  if (useHands && typeof Hands === "undefined") {
    stream.getTracks().forEach((t) => t.stop());
    els.predMeta.textContent = "MediaPipe Hands failed to load (needs internet once for CDN).";
    return;
  }
  if (!useHands && typeof Holistic === "undefined") {
    stream.getTracks().forEach((t) => t.stop());
    els.predMeta.textContent = "MediaPipe Holistic failed to load (needs internet once for CDN).";
    return;
  }

  els.predMeta.textContent = `Using: ${activeLabel} · loading ${useHands ? "Hands" : "Holistic"}…`;
  state.smoothHand = null;

  let tracker;
  if (useHands) {
    tracker = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    tracker.setOptions({
      maxNumHands: 1,
      modelComplexity: 0,
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.5,
    });
    state.trackerType = "hands";
  } else {
    tracker = new Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });
    tracker.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      refineFaceLandmarks: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    state.trackerType = "holistic";
  }
  tracker.onResults(onResults);
  state.tracker = tracker;

  let busy = false;
  const tick = async () => {
    if (!state.cameraOn) return;
    if (!busy && els.video.readyState >= 2) {
      busy = true;
      try {
        await tracker.send({ image: els.video });
      } catch (e) {
        console.warn(e);
      }
      busy = false;
    }
    state._raf = requestAnimationFrame(tick);
  };
  state.cameraOn = true;
  state.camera = {
    stop() {
      cancelAnimationFrame(state._raf || 0);
      stream.getTracks().forEach((t) => t.stop());
    },
  };
  state._raf = requestAnimationFrame(tick);

  els.btnCamera.textContent = "Stop camera";
  if (scoreCameraLabel(activeLabel) >= 0) {
    els.camBadge.textContent = "Laptop camera";
    els.camBadge.className = "badge badge-ok";
    els.predMeta.textContent = `Using: ${activeLabel} · live coach on`;
  }
  if (useHands) setLessonPhase(state.lessonPhase === "done" ? "match" : state.lessonPhase || "match");
}

function stopCamera() {
  if (state._raf) cancelAnimationFrame(state._raf);
  if (state.camera?.stop) state.camera.stop();
  const stream = els.video.srcObject;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  els.video.srcObject = null;
  state.cameraOn = false;
  state.camera = null;
  state.tracker = null;
  state.trackerType = null;
  state.smoothHand = null;
  state._raf = 0;
  els.btnCamera.textContent = "Start camera";
  els.camBadge.textContent = "Camera off";
  els.camBadge.className = "badge";
  seqBuf.clear();
  debouncer.reset();
}

function wireUi() {
  document.querySelectorAll("[data-go]").forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.getAttribute("data-go")));
  });

  const guestBtn = $("btnGuestEnter");
  if (guestBtn) {
    guestBtn.addEventListener("click", async () => {
      if (!state.user) {
        setSession("guest", { id: "guest", name: "Guest", email: "guest@local" });
      }
      showView("app");
      await bootApp();
    });
  }

  const loginForm = $("loginForm");
  const signupForm = $("signupForm");
  if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const errEl = $("loginError");
      if (errEl) {
        errEl.hidden = true;
        errEl.textContent = "";
      }
      const fd = new FormData(loginForm);
      try {
        const data = await api("/api/auth/login", {
          method: "POST",
          body: JSON.stringify({
            email: String(fd.get("email") || "").trim(),
            password: String(fd.get("password") || ""),
          }),
        });
        setSession(data.token, data.user);
        showView("app");
        await bootApp();
      } catch (err) {
        if (errEl) {
          errEl.textContent = err.message || "Login failed";
          errEl.hidden = false;
        }
      }
    });
  }
  if (signupForm) {
    signupForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const errEl = $("signupError");
      if (errEl) {
        errEl.hidden = true;
        errEl.textContent = "";
      }
      const fd = new FormData(signupForm);
      try {
        const data = await api("/api/auth/register", {
          method: "POST",
          body: JSON.stringify({
            name: String(fd.get("name") || "").trim(),
            email: String(fd.get("email") || "").trim(),
            password: String(fd.get("password") || ""),
          }),
        });
        setSession(data.token, data.user);
        showView("app");
        await bootApp();
      } catch (err) {
        if (errEl) {
          errEl.textContent = err.message || "Signup failed";
          errEl.hidden = false;
        }
      }
    });
  }
  const btnLogout = $("btnLogout");
  if (btnLogout) {
    btnLogout.addEventListener("click", async () => {
      try {
        await api("/api/auth/logout", { method: "POST", body: "{}" });
      } catch (_) {}
      clearSession();
      stopCamera();
      showView("home");
    });
  }

  document.querySelectorAll(".mode-btn").forEach((b) => {
    b.addEventListener("click", () => setMode(b.dataset.mode));
  });
  document.querySelectorAll("[data-learn-tab]").forEach((b) => {
    b.addEventListener("click", () => setLearnTab(b.dataset.learnTab));
  });
  document.querySelectorAll("[data-dict-cat]").forEach((b) => {
    b.addEventListener("click", () => {
      state.dictCat = b.dataset.dictCat;
      document.querySelectorAll("[data-dict-cat]").forEach((x) => x.classList.toggle("is-active", x === b));
      loadDictionary();
    });
  });
  els.dictSearch.addEventListener("input", () => loadDictionary());
  els.btnCamera.addEventListener("click", () => {
    startCamera().catch((err) => {
      console.error(err);
      els.predMeta.textContent = `Camera error: ${err?.message || err}`;
      els.camBadge.textContent = "Camera error";
      els.camBadge.className = "badge badge-warn";
    });
  });
  if (els.cameraSelect) {
    els.cameraSelect.addEventListener("change", async () => {
      const id = els.cameraSelect.value;
      if (id) localStorage.setItem(CAMERA_STORE_KEY, id);
      if (state.cameraOn) {
        stopCamera();
        await startCamera();
      }
    });
  }
  els.btnListen.addEventListener("click", listenVoice);
  els.btnQuizNext.addEventListener("click", nextQuiz);
  if (els.btnNextSign) {
    els.btnNextSign.addEventListener("click", () => setTarget(nextSignKey(state.target)));
  }
  if (els.btnReplayGuide) {
    els.btnReplayGuide.addEventListener("click", () => {
      state.hold.reset();
      updateHoldRing(0);
      setLessonPhase("guide");
      applyStepUnlock(0);
      els.coachTip.textContent = "Replay: follow the checklist, then match in the camera.";
    });
  }
  if (els.btnSentenceGotIt) {
    els.btnSentenceGotIt.addEventListener("click", () => {
      if (state.learnTab !== "sentences") return;
      advanceSentenceToken();
    });
  }
  if (els.btnSentenceNext) {
    els.btnSentenceNext.addEventListener("click", () => {
      const sent = currentSentence();
      if (!sent || !state.sentences.length) return;
      const i = state.sentences.findIndex((s) => s.id === sent.id);
      const next = state.sentences[(i + 1) % state.sentences.length];
      selectSentence(next.id);
    });
  }
  els.btnClearTokens.addEventListener("click", () => {
    sentBuf.tokens = [];
    state.tokens = [];
    renderTokens();
  });
  els.btnFlush.addEventListener("click", () => {
    const tokens = sentBuf.flush() || [...state.tokens];
    state.tokens = [];
    renderTokens();
    polishAndSpeak(tokens);
  });
  document.querySelectorAll("[data-emergency]").forEach((btn) => {
    btn.addEventListener("click", () => triggerEmergency(btn.getAttribute("data-emergency")));
  });
}

async function waitForApi() {
  const splash = document.getElementById("bootSplash");
  const status = document.getElementById("bootStatus");
  for (let i = 0; i < 40; i++) {
    try {
      const h = await api("/api/health");
      if (h.ok) {
        if (status) status.textContent = "Ready";
        if (splash) splash.classList.add("is-hidden");
        return true;
      }
    } catch {
      if (status) status.textContent = `Starting engine… (${i + 1})`;
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  if (status) status.textContent = "Engine slow — retrying in app…";
  if (splash) splash.classList.add("is-hidden");
  return false;
}

let appBooted = false;
async function bootApp() {
  if (appBooted) {
    await refreshHealth();
    return;
  }
  appBooted = true;
  loadMastery();
  try {
    const refData = await api("/api/references");
    const n = loadRefs(refData.landmarks || {});
    console.info(`Coach refs loaded: ${n}`);
  } catch (err) {
    console.warn("Could not load coach refs", err);
  }
  try {
    const ok = loadStaticModel(await api("/api/static-model"));
    console.info(`Trained A–Z model loaded: ${ok}`);
  } catch (err) {
    console.warn("Trained A–Z model unavailable — template scoring only", err);
  }
  if (els.holdRingArc) {
    els.holdRingArc.style.strokeDasharray = `${HOLD_CIRC}`;
    els.holdRingArc.style.strokeDashoffset = `${HOLD_CIRC}`;
  }
  buildTargetPicker();
  await refreshCameraList(false);
  await setTarget("A");
  nextQuiz();
  renderTokens();
  setMode("learn");
  setLearnTab("practice");
  setLessonPhase("guide");
  await refreshHealth();
  await loadSignsIndex();
  await loadSentences();
  await loadDictionary();
}

async function main() {
  wireUi();
  await waitForApi();
  const ok = await restoreSession();
  if (ok) {
    showView("app");
    await bootApp();
  } else {
    showView("home");
  }
}

main();
