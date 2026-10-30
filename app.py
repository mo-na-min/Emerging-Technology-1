import os
import json
import sys
import time
from collections import deque

import cv2
import numpy as np
import joblib
import torch

from config import (
    SEQUENCE_LENGTH, STATIC_MODEL_PATH, DYNAMIC_MODEL_PATH,
    DYNAMIC_VOCAB_PATH, HISTORY_DIR, HISTORY_LOG_PATH,
    MODEL_SEES_MIRRORED, MOTION_THRESHOLD,
)
from data_loader import normalize_landmarks
from hand_tracker import HandTracker, draw_hand
from arbitration import ArbitrationEngine
from models import DynamicGRUClassifier
from history_manager import HistoryLogger


def load_models():
    """Loads Static and Dynamic models gracefully."""
    static_clf = None
    dynamic_clf = None
    dynamic_vocab = []

    if os.path.exists(STATIC_MODEL_PATH):
        try:
            static_clf = joblib.load(STATIC_MODEL_PATH)
            # n_jobs=-1 spawns worker threads on every single-sample predict_proba -> laggy
            if hasattr(static_clf, "n_jobs"):
                static_clf.n_jobs = 1
            print("[INFO] Static model loaded successfully.")
        except Exception as e:
            print(f"[WARN] Failed to load static model: {e}")
    else:
        print("[WARN] Static model file missing. Running without static recognition.")

    if os.path.exists(DYNAMIC_MODEL_PATH) and os.path.exists(DYNAMIC_VOCAB_PATH):
        try:
            with open(DYNAMIC_VOCAB_PATH, 'r') as f:
                dynamic_vocab = json.load(f)

            dynamic_clf = DynamicGRUClassifier(
                input_size=63, hidden_size=64, num_layers=2,
                num_classes=len(dynamic_vocab)
            )
            dynamic_clf.load_state_dict(
                torch.load(DYNAMIC_MODEL_PATH, map_location=torch.device('cpu')))
            dynamic_clf.eval()
            print(f"[INFO] Dynamic GRU model loaded. Vocabulary: {dynamic_vocab}")
        except Exception as e:
            print(f"[WARN] Failed to load dynamic model: {e}")
            dynamic_clf = None
    else:
        print("[INFO] Dynamic model not present. App running in Tier-1-only mode.")

    return static_clf, dynamic_clf, dynamic_vocab


def put_fitted_text(img, text, org, max_width, scale, color, thickness=2,
                    font=cv2.FONT_HERSHEY_DUPLEX):
    """Draws text, shrinking the font scale until it fits max_width."""
    while scale > 0.3:
        (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
        if tw <= max_width:
            break
        scale -= 0.05
    cv2.putText(img, text, org, font, scale, color, thickness)


def render_side_panel(panel, label, tag, confidence, motion_state, motion_value,
                      text_builder_str, fps):
    """Side panel sized to fit a 480px-high frame."""
    h, w, _ = panel.shape
    inner = w - 40
    panel[:] = (30, 30, 30)

    cv2.putText(panel, "FSL TRANSLATOR", (20, 32), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(panel, f"{fps:.0f} fps", (w - 70, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    cv2.line(panel, (20, 42), (w - 20, 42), (100, 100, 100), 1)

    cv2.putText(panel, "CURRENT SIGN:", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(panel, str(tag), (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
    put_fitted_text(panel, str(label), (20, 135), inner, 1.1, (255, 255, 255))

    cv2.putText(panel, f"Confidence: {int(confidence * 100)}%", (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    bar_width = int(inner * min(max(confidence, 0.0), 1.0))
    cv2.rectangle(panel, (20, 175), (w - 20, 188), (60, 60, 60), -1)
    cv2.rectangle(panel, (20, 175), (20 + bar_width, 188), (0, 255, 0), -1)

    cv2.putText(panel, f"Status: {motion_state}", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(panel, f"Motion: {motion_value:.3f} (thr {MOTION_THRESHOLD:.3f})", (20, 237),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)
    cv2.line(panel, (20, 252), (w - 20, 252), (100, 100, 100), 1)

    cv2.putText(panel, "TEXT BUILDER:", (20, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.rectangle(panel, (20, 288), (w - 20, 345), (45, 45, 45), -1)
    tail = text_builder_str[-15:]
    cv2.putText(panel, tail, (28, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.putText(panel, "Shortcuts:", (20, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(panel, " 'c' - Clear text", (20, 408), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.putText(panel, " 'q' - Quit", (20, 428), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)


def draw_subtitle_bar(frame, text, live_label):
    """Live-subtitle style overlay at the bottom of the camera view."""
    h, w = frame.shape[:2]
    bar_h = 44
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    shown = text[-40:] if text else ""
    if live_label:
        shown = (shown + "  [" + live_label + "]") if shown else "[" + live_label + "]"
    put_fitted_text(frame, shown, (12, h - 14), w - 24, 0.8, (255, 255, 255), 2,
                    font=cv2.FONT_HERSHEY_SIMPLEX)


def main():
    static_clf, dynamic_clf, dynamic_vocab = load_models()
    if static_clf is None and dynamic_clf is None:
        print("[ERROR] No models loaded. Train them first (train_static.py / train_dynamic.py).")
        sys.exit(1)

    # Real 21-point tracking or nothing - no contour fallback.
    try:
        tracker = HandTracker("video")
    except Exception as e:
        print(f"[ERROR] Hand tracker failed to start: {e}")
        print("Run `python hand_tracker.py` to diagnose.")
        sys.exit(1)
    print("[OK] MediaPipe HandLandmarker (Tasks API) ready.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam feed. Please check video device permissions.")
        return

    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    arbitrator = ArbitrationEngine()
    logger = HistoryLogger(HISTORY_DIR, HISTORY_LOG_PATH)

    MISS_GRACE = 5          # tolerate a few dropped detections before resetting state
    misses = 0
    label, tag, conf, status = "No hand detected", "[N/A]", 0.0, "Searching hand..."
    motion_value = 0.0
    fps, last_t = 0.0, time.time()

    cv2.namedWindow("Filipino Sign Language Translator", cv2.WINDOW_NORMAL)

    while True:
        ret, raw = cap.read()
        if not ret:
            print("[ERROR] Camera stream disconnected.")
            break

        display = cv2.flip(raw, 1)                       # mirrored preview for the user
        model_frame = display if MODEL_SEES_MIRRORED else raw

        pts = tracker.detect(model_frame)

        if pts is not None:
            misses = 0

            draw_pts = pts.copy()
            if not MODEL_SEES_MIRRORED:
                draw_pts[:, 0] = 1.0 - draw_pts[:, 0]    # map raw-frame coords onto mirrored preview
            draw_hand(display, draw_pts)

            norm_vec = normalize_landmarks(pts)
            sequence_buffer.append(norm_vec)
            motion_value = arbitrator.update_motion(pts)

            static_pred, dynamic_pred = None, None

            if static_clf is not None:
                probs = static_clf.predict_proba(norm_vec.reshape(1, -1))[0]
                best = int(np.argmax(probs))
                static_pred = (str(static_clf.classes_[best]), float(probs[best]))

            if (dynamic_clf is not None and arbitrator.in_motion
                    and len(sequence_buffer) == SEQUENCE_LENGTH):
                seq_tensor = torch.FloatTensor(np.array(sequence_buffer)).unsqueeze(0)
                with torch.no_grad():
                    out = dynamic_clf(seq_tensor)
                    probs = torch.softmax(out, dim=1)[0]
                    best = int(torch.argmax(probs).item())
                    dynamic_pred = (dynamic_vocab[best], float(probs[best].item()))

            label, tag, conf, status = arbitrator.arbitrate_and_smooth(
                static_pred, dynamic_pred, allow_dynamic=dynamic_clf is not None)

            if arbitrator.process_dwell(label, tag, conf):
                logger.log_event(display, label, tag, conf)
        else:
            misses += 1
            if misses > MISS_GRACE:
                sequence_buffer.clear()
                arbitrator.reset()
                motion_value = 0.0
                label, tag, conf, status = "No hand detected", "[N/A]", 0.0, "Searching hand..."

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_t, 1e-3))
        last_t = now

        # UI
        text_builder_str = "".join(arbitrator.committed_text_builder)
        live = label if label not in ("No hand detected", "No prediction", "Analyzing...") else ""
        draw_subtitle_bar(display, text_builder_str, live if conf > 0 else "")

        h, w, _ = display.shape
        panel = np.zeros((h, int(w * 0.35), 3), dtype=np.uint8)
        render_side_panel(panel, label, tag, conf, status, motion_value, text_builder_str, fps)

        cv2.imshow("Filipino Sign Language Translator", np.hstack((display, panel)))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            arbitrator.committed_text_builder.clear()

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()