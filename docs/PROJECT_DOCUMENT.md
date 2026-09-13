# Real-Time Two-Way Sign Language & Communication Bridge
## Complete Project Document (Brief · Stack · Research · Comparison · Team Plan · Report)

**Team size:** 3 members  
**Document purpose:** Single source of truth for what we are building, what tools we use, what to research online, how similar projects compare, how to split work, and what to put in the academic/project report.  
**Date:** August 2026

---

## Table of Contents
1. [Project Goal & Problem Statement](#1-project-goal--problem-statement)
2. [High-Impact Differentiators](#2-high-impact-differentiators)
3. [What We Are Using to Build This (Tech Stack)](#3-what-we-are-using-to-build-this-tech-stack)
4. [System Architecture](#4-system-architecture)
5. [Key Modules](#5-key-modules)
6. [What We Need to Learn / Research Online](#6-what-we-need-to-learn--research-online)
7. [Comparison of Similar Projects & Approaches](#7-comparison-of-similar-projects--approaches)
8. [Recommended Choices (Best Result Path)](#8-recommended-choices-best-result-path)
9. [Team Task Division (3 Members)](#9-team-task-division-3-members)
10. [Project Directory Structure](#10-project-directory-structure)
11. [Implementation Roadmap & Milestones](#11-implementation-roadmap--milestones)
12. [Dataset Plan](#12-dataset-plan)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [Risks & Mitigations](#14-risks--mitigations)
15. [Report Outline (Submit This Structure)](#15-report-outline-submit-this-structure)
16. [How to Run the Scaffold](#16-how-to-run-the-scaffold)
17. [References & Research Links to Explore](#17-references--research-links-to-explore)

---

## 1. Project Goal & Problem Statement

### Goal
Build an **end-to-end, real-time** application that:
1. Recognizes **continuous dynamic sign gestures** from a webcam and turns them into **spoken natural language**.
2. Accepts **spoken voice** from a microphone and shows matching **sign animations / GIFs / text** so hearing and Deaf/Hard-of-Hearing users can talk both ways.

### Problem
Most student demos only do **static finger-spelling (A–Z)**. Real conversation uses **words and short phrases** that move over time. Raw sign glosses also lack English grammar (`I go hospital` → should become `I am going to the hospital.`). We solve both with a **temporal sequence model** plus an **LLM sentence polisher**, and close the loop with **speech-to-sign visualization**.

### Target users (demo context)
- Deaf / Hard-of-Hearing users communicating with hearing users
- Emergency / quick-help situations (high-priority phrases)
- Classroom / hospital / service-desk accessibility prototype

---

## 2. High-Impact Differentiators

| # | Differentiator | Why it matters |
|---|----------------|----------------|
| 1 | **Dynamic temporal gestures** | Words like “Thank you”, “Need help”, “Water” need multi-frame motion, not one still image |
| 2 | **Two-way loop** | Sign→Speech **and** Speech→Sign (not one-direction only) |
| 3 | **LLM sentence reconstruction** | Fixes missing grammar/connectives common in gloss streams |
| 4 | **Emergency / Quick-Action mode** | Hotkeys for “Emergency / Call 108”, “Medical Help” with visual flash + priority audio |

### Pipelines
**Sign → Speech**  
Webcam → MediaPipe keypoints → 30-frame buffer → LSTM classifier → token buffer → LLM polish → TTS

**Speech → Sign**  
Microphone → SpeechRecognition / Whisper → keyword match → GIF / animation viewer

---

## 3. What We Are Using to Build This (Tech Stack)

### Locked choices for this repository

| Layer | Technology | Role |
|-------|------------|------|
| Language | **Python 3.10+** | Entire backend + ML + UI scripts |
| Vision | **OpenCV** | Webcam capture, flip, overlay, display |
| Landmarks | **MediaPipe Holistic** (hands + pose) | 21 landmarks/hand + selected upper-body joints |
| Sequence model | **PyTorch LSTM** | Classify dynamic gesture windows `(30 × features)` |
| UI dashboard | **Streamlit** | Live video, buffers, chat, emergency buttons |
| Sign→Voice | **pyttsx3** (offline) / optional **gTTS** | Speak polished sentences |
| Voice→Text | **SpeechRecognition** (+ optional **OpenAI Whisper**) | Transcribe microphone |
| Grammar LLM | **Rule fallback** + **Ollama** (local) or **Groq API** | Gloss → natural English |
| Data format | **NumPy `.npy` sequences** | Custom dataset per gesture label |
| Config / secrets | **python-dotenv** | `GROQ_API_KEY` etc. |

### Optional upgrades (after MVP works)
- **1D-CNN + LSTM** or **Transformer** encoder for better accuracy
- **faster-whisper** for local STT without Google cloud
- **PyQt6** if we need a desktop app instead of browser UI
- Public datasets: **WLASL**, **MS-ASL**, **INCLUDE** (Indian Sign Language) for pretraining ideas

### What we are *not* using for MVP (on purpose)
- Full 3D body MoCap suites (too heavy for laptop demo)
- Training huge video CNNs from raw pixels (slow; keypoints are enough for a strong demo)
- Paid cloud vision APIs as the core recognizer (we keep the pipeline local/explainable)

---

## 4. System Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Webcam    │────▶│ Landmark Extract │────▶│ 30-frame Buffer │
└─────────────┘     │  (normalize)     │     └────────┬────────┘
                    └──────────────────┘              │
                                                      ▼
                                            ┌─────────────────┐
                                            │  LSTM Classifier │
                                            │ + Debounce (>0.85│
                                            │   for 10 frames) │
                                            └────────┬────────┘
                                                     │ tokens
                                                     ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│    TTS      │◀────│  LLM Polisher    │◀────│ Sentence Buffer │
│  pyttsx3    │     │ Ollama/Groq/rule │     │ idle 2s / EOS   │
└─────────────┘     └──────────────────┘     └─────────────────┘

┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Microphone  │────▶│ STT (SR/Whisper) │────▶│ Keyword → GIFs  │
└─────────────┘     └──────────────────┘     │ assets/signs/   │
                                             └─────────────────┘

Emergency hotkeys ──▶ UI flash + priority TTS + forced gloss token
```

### Feature vector (per frame)
- Left hand 21×3 + right hand 21×3 = **126**
- Selected pose joints (nose, shoulders, elbows, wrists) ≈ **21**
- **Total ≈ 147 floats / frame**
- Normalized: wrist-relative (hands), shoulder-midpoint (pose), scaled by bbox size → **distance invariance**

---

## 5. Key Modules

### Module A — Landmark Extraction & Normalization
**File:** `src/tracking/landmarks.py`  
- MediaPipe Holistic  
- Wrist-relative coords: `(x - x_wrist, y - y_wrist, z - z_wrist)`  
- Scale by hand / pose extent so near vs far camera still match  

### Module B — Real-Time Sequence Prediction
**Files:** `src/model/buffer.py`, `src/model/lstm_model.py`  
- Circular / deque buffer of last **30** frames  
- Debounce: confidence **> 0.85** for **≥ 10** consecutive frames before emitting a word  
- Cooldown to avoid repeating the same word instantly  

### Module C — Sentence Buffer & NLP Post-Processing
**File:** `src/nlp/sentence.py`  
- Accumulate tokens  
- Flush on **EOS gesture** or **~2 s idle**  
- Pass to grammar engine → TTS  

### Module D — Two-Way Reverse Translation
**File:** `src/audio/io.py`  
- Mic → text  
- Keyword index over `assets/signs/*.gif` (e.g. `thank_you.gif`, `water.gif`)  

### Emergency Mode
**File:** `src/ui/emergency.py`  
- Quick actions: Emergency / Call 108, Medical Help, Need Help  
- Visual flash + blocking high-priority speech  

### Entry scripts
| Script | Purpose |
|--------|---------|
| `data_collection.py` | Record labeled keypoint sequences from webcam |
| `train_model.py` | Train LSTM, save `models/sign_lstm.pt` |
| `app.py` | Streamlit two-way dashboard |

---

## 6. What We Need to Learn / Research Online

Use this as the **research checklist** before / while coding. Each teammate should take notes and paste links into a shared folder.

### A. Computer Vision & MediaPipe
- [ ] How MediaPipe **Hands** vs **Holistic** differ (hands-only vs pose+face+hands)
- [ ] Landmark indices (wrist = 0, fingertips, etc.)
- [ ] Why **normalization** matters (scale, translation, sometimes rotation)
- [ ] Real-time FPS tips on CPU vs GPU

**Search queries:**  
`MediaPipe Holistic Python tutorial`, `sign language mediapipe landmarks normalization`

### B. Sequence models for gestures
- [ ] Difference: **static image CNN** vs **LSTM/GRU/Transformer on keypoints**
- [ ] Sliding window / temporal convolution ideas
- [ ] Overfitting on small custom datasets; data augmentation (speed jitter, noise, mirror)
- [ ] Confidence thresholding and **temporal smoothing / voting**

**Search queries:**  
`LSTM sign language recognition mediapipe`, `dynamic hand gesture recognition pytorch`, `WLASL benchmark`

### C. Sign language linguistics (important for report quality)
- [ ] Gloss notation vs spoken English
- [ ] Why signs omit articles (“the”, “a”) and tense markers
- [ ] Difference **ASL / ISL / BSL** — pick **one** signing system for the demo and state it clearly
- [ ] Continuous signing vs isolated word recognition

**Search queries:**  
`ASL gloss to English`, `Indian Sign Language INCLUDE dataset`, `continuous sign language recognition survey`

### D. Speech systems
- [ ] Offline TTS (`pyttsx3`) vs cloud TTS
- [ ] Whisper model sizes (`tiny`/`base`/`small`) vs latency
- [ ] Microphone permissions and noise handling on Windows

### E. LLMs for grammar repair
- [ ] Local **Ollama** setup (Llama 3.2 / Gemma)
- [ ] **Groq** free-tier API for fast Llama inference
- [ ] Prompt design: “only use given tokens; do not invent facts”

### F. HCI / accessibility
- [ ] Emergency UX: high contrast flash, clear audio, no silent failure
- [ ] Latency budget: aim **< 300–500 ms** after gesture ends for token; sentence after idle pause is OK
- [ ] Ethics: consent for camera, no medical claims, 108 is demo text not a real dialer unless integrated carefully

### G. Papers / surveys to skim (for report Chapter 2)
Search Google Scholar / arXiv for:
- Continuous Sign Language Recognition (CSLR)
- Sign Language Translation (SLT) — gloss or text targets
- MediaPipe + LSTM student projects (many on GitHub; use for comparison, not copy)

---

## 7. Comparison of Similar Projects & Approaches

Research these **categories** on GitHub, YouTube, IEEE papers, and Kaggle. Fill the “Our notes” column as a team.

### 7.1 Approach comparison

| Approach | How it works | Pros | Cons | Fit for us |
|----------|--------------|------|------|------------|
| **Static CNN on hand crop (A–Z)** | One frame → letter | Easy, many tutorials | Not real words; weak for conversation | ❌ Demo-only, not our goal |
| **MediaPipe + classical ML (SVM/RF)** | Landmarks → classifier | Fast train, simple | Weak on dynamic phrases | ⚠ Backup baseline |
| **MediaPipe + LSTM/GRU (our plan)** | 30-frame keypoint seq → word | Good accuracy/speed balance; laptop-friendly | Needs custom data collection | ✅ **Primary** |
| **Raw video 3D CNN / I3D** | Learn from pixels | Strong on big datasets | Needs GPU, huge data, slow iterate | ⚠ Future / research only |
| **Transformer on keypoints** | Attention over time | Strong long context | More data + tuning | ⚠ Phase-2 upgrade |
| **Commercial APIs** | Cloud sign recognition | Fast to demo | Cost, privacy, less “we built it” | ❌ Not for core credit |

### 7.2 Feature comparison vs typical student projects

| Feature | Typical A–Z demo | Many GitHub “sign LSTM” repos | **Our project** |
|---------|------------------|-------------------------------|-----------------|
| Dynamic words/phrases | Rare | Sometimes | **Yes** |
| Debounced real-time output | Rare | Sometimes | **Yes** |
| LLM grammar polish | Almost never | Almost never | **Yes** |
| Speech → Sign visuals | Rare | Rare | **Yes** |
| Emergency quick actions | No | No | **Yes** |
| End-to-end Streamlit UI | Sometimes | Sometimes | **Yes** |

### 7.3 Dataset / language comparison (choose one primary)

| Resource | Language | Notes | Use |
|----------|----------|-------|-----|
| **Custom webcam set** (our `data_collection.py`) | Whatever team signs | Best control; required for demo reliability | **Must have** |
| INCLUDE | Indian Sign Language | Good if team targets ISL | Optional pretrain / vocab ideas |
| WLASL | ASL word clips | Large; different domain (RGB video) | Research comparison |
| MS-ASL | ASL | Large scale | Research comparison |
| Finger-spelling sets | Letters | Easy but off-goal | Baseline only |

### 7.4 UI / product comparison

| UI choice | Pros | Cons | Decision |
|-----------|------|------|----------|
| **Streamlit** | Fast to build, good for demo day | Webcam loops are a bit awkward | ✅ MVP |
| PyQt6 / Electron | Polished desktop feel | Slower to develop | Later |
| Web (React + backend) | Nice UX | Split stack, more ops | Later |

### 7.5 LLM backend comparison

| Backend | Cost | Privacy | Latency | Decision |
|---------|------|---------|---------|----------|
| **Rule-based map** | Free | Local | Instant | ✅ Always keep as fallback |
| **Ollama local** | Free | Local | Depends on PC | ✅ Preferred offline |
| **Groq API** | Free tier / cheap | Cloud | Very fast | ✅ Preferred if Wi-Fi OK |
| OpenAI GPT | Paid | Cloud | Fast | Optional |

---

## 8. Recommended Choices (Best Result Path)

To get the **best demo result** with 3 people in limited time:

1. **Freeze vocabulary early** — start with **8–12 signs** only, e.g.  
   `hello`, `thank_you`, `yes`, `no`, `water`, `need_help`, `hospital`, `pain`, `family`, `emergency`, `medical_help`, `EOS`
2. **Collect ≥ 40 samples per class** (different speeds, distances, both teammates signing)
3. **Train LSTM on keypoints** (not raw video) first; measure val accuracy before UI polish
4. **Debounce aggressively** — better few correct words than flickering garbage
5. **LLM polish with tight prompt** — never invent hospital names or phone numbers
6. **Speech→Sign** = keyword GIFs first (don’t try generative avatar in v1)
7. **One signing system** (ASL *or* ISL) stated in the report — don’t mix casually
8. **Emergency buttons** always work even if the vision model fails

---

## 9. Team Task Division (3 Members)

Assign real names in the blanks. Work in **parallel tracks** that meet at integration milestones.

| Role | Member | Owns | Deliverables |
|------|--------|------|--------------|
| **Member A — Vision & Data** | _____________ | Webcam, MediaPipe, normalization, data collection UX, dataset quality | Module A; `data_collection.py`; `data/raw/`; landmark overlay quality |
| **Member B — ML & Training** | _____________ | Buffer, LSTM, training script, metrics, model export | Module B; `train_model.py`; `models/sign_lstm.pt`; accuracy table for report |
| **Member C — App, Audio, NLP, Emergency** | _____________ | Streamlit UI, TTS/STT, LLM polish, GIF matcher, emergency mode | Modules C–D; `app.py`; `assets/signs/`; demo script |

### Shared responsibilities (all three)
- Agree on **vocab list** and **signing system (ASL/ISL)** in week 1
- Weekly 30-min integration sync
- Contribute 1 page of **related-work comparison** notes (Section 7)
- Joint dry-run of final demo + report chapters

### Suggested weekly split (6-week example)

| Week | Member A | Member B | Member C |
|------|----------|----------|----------|
| 1 | Research MediaPipe; normalize prototype | Research LSTM gesture papers; define tensor shapes | Research Streamlit + TTS/STT; sketch UI wireframe |
| 2 | Finish `data_collection.py`; collect 50% data | Scaffold `SignLSTM` + train smoke test | Sentence buffer + rule polisher; emergency buttons |
| 3 | Finish dataset; mirror augmentation | Full train + tune threshold/debounce | Wire prediction into Streamlit; chat history |
| 4 | Hard-example recollect (confused classes) | Confusion matrix; improve model | Ollama/Groq polish; Whisper optional |
| 5 | Lighting / distance robustness tests | Export final checkpoint; docs for metrics | Speech→Sign GIFs; polish UI dark theme |
| 6 | Demo support + report Ch.3–4 | Report Ch.4–5 (results) | Report Ch.1,6,7 + live demo lead |

### Integration contract (so nobody blocks)
- Feature dim locked: **147** (or update all three together)
- Window locked: **30** frames
- Labels file: `models/labels.json`
- Token names = folder names under `data/raw/` = GIF stems under `assets/signs/`

---

## 10. Project Directory Structure

```
project 2/
├── app.py                 # Streamlit dashboard
├── data_collection.py     # Record keypoint sequences
├── train_model.py         # Train LSTM
├── requirements.txt
├── .env.example
├── README.md
├── docs/
│   └── PROJECT_DOCUMENT.md   # THIS FILE
├── data/
│   ├── raw/<label>/*.npy
│   └── processed/            # optional future exports
├── models/
│   ├── sign_lstm.pt
│   └── labels.json
├── assets/
│   └── signs/                # thank_you.gif, water.gif, ...
└── src/
    ├── tracking/
    │   └── landmarks.py      # Module A
    ├── model/
    │   ├── buffer.py         # Module B buffer + debounce
    │   └── lstm_model.py     # LSTM + predictor
    ├── nlp/
    │   └── sentence.py       # Module C
    ├── audio/
    │   └── io.py             # Module D TTS/STT/GIF match
    └── ui/
        └── emergency.py      # Emergency mode
```

---

## 11. Implementation Roadmap & Milestones

| Milestone | Done when… |
|-----------|------------|
| M0 Scaffold | Repo runs; imports work; empty UI opens |
| M1 Landmarks | Overlay stable; `.npy` samples save correctly |
| M2 Dataset | ≥ 8 classes × ≥ 40 clips |
| M3 Model | Val accuracy usable (target ≥ 85% on isolated clips) |
| M4 Live tokens | Debounced words appear in buffer without insane flicker |
| M5 Sentences | Idle flush + polish + TTS |
| M6 Reverse | Mic phrase shows correct GIF(s) |
| M7 Emergency | Buttons flash + speak even with camera off |
| M8 Demo polish | 3-minute script rehearsed; report draft complete |

---

## 12. Dataset Plan

### Initial vocabulary (edit as team agrees)
1. hello  
2. thank_you  
3. yes  
4. no  
5. water  
6. need_help  
7. hospital  
8. pain  
9. family  
10. emergency  
11. medical_help  
12. EOS (end of sentence gesture)

### Collection protocol
- Same camera resolution for all collectors  
- Include: close / medium / slightly far; slow / normal speed  
- Both left- and right-handed signers if possible  
- Reject clips with missing hands for hand-only signs  
- Keep a spreadsheet: label, sample count, who signed, date  

### Command examples
```bash
python data_collection.py --label thank_you --samples 40
python data_collection.py --label water --samples 40
python train_model.py --epochs 30
streamlit run app.py
```

---

## 13. Evaluation Metrics

| Metric | What it measures |
|--------|------------------|
| Per-class accuracy / F1 | Classification quality |
| Confusion matrix | Which signs collide (e.g. thank_you vs hello) |
| False trigger rate | Words emitted when idle |
| End-to-end latency | Gesture end → token; idle → spoken sentence |
| STT keyword hit rate | Voice phrase → correct GIF |
| User demo success | Scripted dialogue completed without restart |

Log these in the report with at least one table and one confusion-matrix figure.

---

## 14. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Tiny dataset → overfitting | More samples; augment; keep vocab small |
| Flickering predictions | Debounce + cooldown + higher threshold |
| Lighting / background fail | Consistent demo backdrop; Holistic confidence checks |
| LLM invents content | Strict prompt + rule fallback |
| Mic fails on demo PC | Pre-record backup phrase OR type-to-GIF fallback |
| MediaPipe version breaks | Pin versions in `requirements.txt` |
| Scope creep (avatars, full ISL) | Park as “Future Work” |

---

## 15. Report Outline (Submit This Structure)

Use this as the **formal project report** skeleton. Expand each section with screenshots, diagrams, and citations from your online research.

### Title page
- Project title: Real-Time Two-Way Sign Language & Communication Bridge  
- Team members (3), mentor/guide, college, date  

### Abstract (150–250 words)
Problem, approach (MediaPipe + LSTM + LLM + two-way audio), key results, conclusion.

### Chapter 1 — Introduction
- 1.1 Background & motivation (accessibility)  
- 1.2 Problem statement  
- 1.3 Objectives  
- 1.4 Scope & limitations  
- 1.5 Differentiators (dynamic signs, two-way, LLM, emergency)  

### Chapter 2 — Literature Survey / Related Work
- Static vs dynamic recognition  
- Keypoint vs RGB video methods  
- Datasets (WLASL, MS-ASL, INCLUDE, custom)  
- Sign language translation & glosses  
- **Comparison table** (copy Section 7, fill with real citations)  
- Gap analysis → why our design  

### Chapter 3 — System Requirements
- Hardware: webcam, mic, laptop CPU/GPU  
- Software: Python 3.10+, packages list  
- Functional / non-functional requirements  

### Chapter 4 — Design & Methodology
- Architecture diagram (Section 4)  
- Module A–D design  
- Normalization math  
- LSTM architecture (layers, input shape)  
- Debounce & sentence buffer logic  
- LLM prompt design  
- Emergency mode UX  

### Chapter 5 — Implementation
- Directory structure  
- Important libraries & why chosen  
- Data collection procedure  
- Training setup (epochs, batch, device)  
- Streamlit UI screenshots  

### Chapter 6 — Results & Discussion
- Dataset stats  
- Accuracy / F1 / confusion matrix  
- Latency observations  
- Failure cases  
- Comparison vs baseline (e.g. SVM on single frame)  

### Chapter 7 — Conclusion & Future Work
- Summary of achievements  
- Future: larger vocab, Transformer, ISL expansion, real calling integration (with ethics), mobile app  

### References
IEEE/APA style from your research links.

### Appendices
- Full vocab list  
- Sample prompts  
- Installation guide  
- Demo script (2–3 minutes)

### Suggested demo script
1. Show live landmarks  
2. Sign: `need_help` + `water` → polished sentence + voice  
3. Speak: “thank you” → GIF  
4. Press Emergency → flash + audio  
5. Show confusion matrix slide  

---

## 16. How to Run the Scaffold

```bash
# 1) Create venv (Windows PowerShell)
cd "c:\Users\yuvra\OneDrive\Desktop\project 2"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) Collect data
python data_collection.py --label thank_you --samples 40

# 3) Train
python train_model.py --epochs 25

# 4) (Optional) LLM
#    Install Ollama and pull a model, OR set GROQ_API_KEY in .env

# 5) Add GIFs into assets/signs/  (thank_you.gif, water.gif, ...)

# 6) Launch app
streamlit run app.py
```

---

## 17. References & Research Links to Explore

> Verify each link while researching; prefer papers + official docs over random blogs.

**Official docs**
- MediaPipe: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker  
- OpenCV Python: https://docs.opencv.org/  
- PyTorch LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html  
- Streamlit camera/input patterns: https://docs.streamlit.io/  
- Ollama: https://ollama.com/  
- Groq API: https://console.groq.com/  
- OpenAI Whisper: https://github.com/openai/whisper  
- pyttsx3: https://pyttsx3.readthedocs.io/  

**Datasets / benchmarks (compare in report)**
- WLASL: https://dxli94.github.io/WLASL/  
- INCLUDE (ISL): search `INCLUDE Indian Sign Language dataset`  
- MS-ASL: search `MS-ASL Microsoft dataset`  

**Survey starting points (Scholar)**
- “Continuous Sign Language Recognition Survey”  
- “Sign Language Translation neural networks”  
- “MediaPipe LSTM Indian Sign Language”  

**Ethics & accessibility**
- W3C WAI / accessibility basics  
- Local emergency-number regulations before any real dial feature  

---

## Appendix A — Decision Log (fill during project)

| Date | Decision | Why | Owner |
|------|----------|-----|-------|
| | ASL vs ISL | | |
| | Vocab v1 list | | |
| | Ollama vs Groq default | | |
| | Streamlit kept for final demo? | | |

## Appendix B — Meeting Notes Template

| Date | Attendees | Done | Blockers | Next |
|------|-----------|------|----------|------|
| | A/B/C | | | |

---

**End of document.**  
This file is the master brief + research guide + comparison framework + team plan + report outline. Update Section 7 tables with real GitHub/paper entries as you research so the final report comparison is evidence-based, not generic.
