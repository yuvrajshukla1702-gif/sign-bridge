# Sign Bridge — Dual-Mode Desktop App

**Mode 1 — Learning Hub:** A–Z / 0–9 dictionary, live pose matcher with %, quiz mode, sentence phrases  
**Mode 2 — Live Conversation:** Sign↔Speech bridge, LLM polish, emergency actions  

Pipeline inspired by [Nick Renotte’s MediaPipe + LSTM action detection](https://github.com/nicknochnack/ActionDetectionforSignLanguage) ([video](https://youtu.be/pDXdlXlaCco)).  
Optional LabelImg / TF Object Detection path is documented in [`docs/TRAINING_PIPELINE.md`](docs/TRAINING_PIPELINE.md).

## Open
Double-click **Sign Bridge** on your Desktop, or run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-app.ps1
```

Then open http://127.0.0.1:8000 if the browser does not open automatically.

**App won't start?** Port 8000 may be stuck. Run `start-app.ps1` — it clears stale servers and restarts cleanly.

```powershell
npm run desktop:live
# or manual API only:
.\.venv\Scripts\python.exe run_webapp.py
```

## Learn Mode tips
1. Start camera (top bar)
2. Use the floating **Visual cue** card — copy that handshape
3. Pick a letter / open **Sentences** for hello, thank you, I love you, …
4. Match the pose — coach tips + hold ring
5. Try **Challenge** or browse **Library**

Regenerate cue images:

```powershell
python scripts/generate_sign_cue_cards.py
```

Reference landmarks: `data/references/static_landmarks.json`  
(Regenerate templates: `python scripts/generate_static_references.py`)

## Bridge Mode (Nick-style 30-frame clips)
Train dynamic phrases for live recognition:

```powershell
python data_collection.py --label hello --samples 30
python data_collection.py --label thank_you --samples 30
python data_collection.py --label iloveyou --samples 30
python train_model.py --window 30 --epochs 40
```

## Layout
- `web/` — dual-mode UI + live visual cues  
- `backend/main.py` — API (predict, evaluate, dictionary, polish, sentences)  
- `src/engine/` — static + sequence classifiers, feedback, bridge  
- `src/tracking/` — landmark normalization (MediaPipe-aligned)  
- `assets/signs*` — teaching cue cards  
- `docs/TRAINING_PIPELINE.md` — Nick / LabelImg / TFOD notes  
