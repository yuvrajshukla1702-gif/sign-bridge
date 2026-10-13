# Training pipeline notes (Nick Renotte + LabelImg / TFOD)

Sign Bridge follows the **action-recognition** approach popularized by
[Nicholas Renotte’s MediaPipe + LSTM sign language tutorial](https://github.com/nicknochnack/ActionDetectionforSignLanguage)
([YouTube walkthrough](https://youtu.be/pDXdlXlaCco)):

1. Extract body/hand landmarks each frame (MediaPipe Holistic / Hands)
2. Collect fixed-length sequences (**30 frames**) per class
3. Train an LSTM / dense sequence model
4. Debounce predictions in real time (confidence + consecutive frames)

That matches our stack more closely than classic **bounding-box** object detection.

## What we use (default)

| Step | Tool in this repo |
|------|-------------------|
| Landmarks | MediaPipe via `src/tracking/` + browser Hands/Holistic |
| Collect clips | `python data_collection.py --label thank_you --samples 30` (defaults to **30 frames**) |
| Train | `python train_model.py` → `models/sign_lstm.pt` |
| Live bridge | Learning Hub + Live Conversation in `web/` |
| Visual cues | `scripts/generate_sign_cue_cards.py` → `assets/signs*` |

## Optional path: LabelImg + TensorFlow Object Detection

If you later want **box-based** detection (draw rectangles around the signer)
instead of landmarks, use:

- [LabelImg](https://github.com/HumanSignal/labelImg) — draw boxes, export Pascal VOC / YOLO  
  *(archived; Label Studio is the modern successor)*
- [TensorFlow Object Detection install guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)
- Nick’s TFOD scaffolding: [RealTimeObjectDetection](https://github.com/nicknochnack/RealTimeObjectDetection)

Rough workflow:

1. Capture photos/video frames of each sign
2. Label with LabelImg / Label Studio
3. Train an SSD / Faster-R-CNN style detector
4. Run webcam inference and map class IDs → spoken glosses

**We intentionally keep MediaPipe+LSTM as the primary path** — it’s lighter on
CPU, works well for dynamic signs (thank you, hello), and already powers
Learning Hub pose matching.

## Suggested vocabulary (aligned with common demos)

- `hello`, `thank_you`, `iloveyou`, `yes`, `no`, `water`, `need_help`, …

Collect ~30 samples × 30 frames each before training:

```powershell
python data_collection.py --label hello --samples 30
python data_collection.py --label thank_you --samples 30
python data_collection.py --label iloveyou --samples 30
python train_model.py --epochs 40 --window 30
```

## Visual cue cards

Regenerate teaching images anytime:

```powershell
python scripts/generate_sign_cue_cards.py
```

Cards appear as a floating **Visual cue** panel on the Learning Hub camera,
plus Library / Sentences thumbnails.
