# Sign Bridge — Dual-Mode Desktop App

**Mode 1 — Learning Hub:** A–Z / 0–9 dictionary, live pose matcher with %, quiz mode  
**Mode 2 — Live Conversation:** Sign↔Speech bridge, LLM polish, emergency actions  

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
2. Pick a letter/number
3. Match the pose — coach tips + score meter
4. Try **Quiz** or browse **Dictionary**

Reference landmarks: `data/references/static_landmarks.json`  
(Regenerate templates: `python scripts/generate_static_references.py`)

## Bridge Mode
Train dynamic phrases for live recognition:

```powershell
python data_collection.py --label thank_you --samples 40
python train_model.py
```

## Layout
- `web/` — dual-mode UI  
- `backend/main.py` — API (predict, evaluate, dictionary, polish)  
- `src/engine/` — static + sequence classifiers, feedback, bridge  
- `src/tracking/extractor.py` — landmark normalization  
- `assets/signs_dict/` — dictionary cards  
