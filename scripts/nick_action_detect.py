"""
Nick Renotte Action Detection prototype (MediaPipe Holistic + LSTM).

From: Action Detection Tutorial.ipynb realtime loop
Weights: models/action.h5  (hello / thanks / iloveyou)

  .\\.venv_nick\\Scripts\\python.exe scripts/nick_action_detect.py --smoke
  .\\.venv_nick\\Scripts\\python.exe scripts/nick_action_detect.py --frames 60
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "_vendor" / "ActionDetectionforSignLanguage"
MODEL_CANDIDATES = [
    ROOT / "models" / "action.h5",
    VENDOR / "action.h5",
]

ACTIONS = np.array(["hello", "thanks", "iloveyou"])
COLORS = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
THRESHOLD = 0.8


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results, mp_drawing, mp_holistic):
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


def extract_keypoints(results):
    pose = (
        np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    lh = (
        np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    rh = (
        np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, lh, rh])


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(
            output_frame,
            actions[num],
            (0, 85 + num * 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output_frame


def build_nick_model():
    from tensorflow.keras.layers import Dense, Input, LSTM
    from tensorflow.keras.models import Sequential, load_model

    model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    if model_path is None:
        raise SystemExit("action.h5 not found")

    print(f"Loading {model_path}")
    model = Sequential(
        [
            Input(shape=(30, 1662)),
            LSTM(64, return_sequences=True, activation="relu"),
            LSTM(128, return_sequences=True, activation="relu"),
            LSTM(64, return_sequences=False, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(ACTIONS.shape[0], activation="softmax"),
        ]
    )
    try:
        model.load_weights(str(model_path))
    except Exception:
        model = load_model(str(model_path), compile=False)
    print("model_ready", model.output_shape)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    model = build_nick_model()
    x = np.zeros((1, 30, 1662), dtype=np.float32)
    res0 = model.predict(x, verbose=0)[0]
    print("smoke_pred", ACTIONS[int(np.argmax(res0))], float(np.max(res0)))
    if args.smoke:
        print("PROTO_OK model_load")
        return

    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        print("WARN: mediapipe 1.x has no solutions — synthetic loop")
        seq = [np.random.randn(1662).astype(np.float32) * 0.01 for _ in range(30)]
        res = model.predict(np.expand_dims(seq, axis=0), verbose=0)[0]
        print("pred", ACTIONS[int(np.argmax(res))], float(np.max(res)))
        print("PROTO_OK synthetic_loop")
        return

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    sequence: deque = deque(maxlen=30)
    sentence = []
    last_pred = "—"
    last_conf = 0.0

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("NO_CAMERA")
        print("PROTO_OK model_load")
        return

    frame_i = 0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results, mp_drawing, mp_holistic)
            sequence.append(extract_keypoints(results))

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(list(sequence), axis=0), verbose=0)[0]
                idx = int(np.argmax(res))
                last_pred = ACTIONS[idx]
                last_conf = float(res[idx])
                if last_conf > THRESHOLD:
                    if not sentence or ACTIONS[idx] != sentence[-1]:
                        sentence.append(ACTIONS[idx])
                    sentence = sentence[-5:]
                image = prob_viz(res, ACTIONS, image, COLORS)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, " ".join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if args.frames > 0:
                frame_i += 1
                if frame_i >= args.frames:
                    print(f"pred={last_pred} conf={last_conf:.3f} sentence={sentence}")
                    print("PROTO_OK")
                    break
            else:
                cv2.imshow("Nick Action Detection", image)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()
    print(f"last={last_pred} conf={last_conf:.3f}")
    print("PROTO_OK")


if __name__ == "__main__":
    main()
