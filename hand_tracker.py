"""
21-point hand tracking via the MediaPipe Tasks API (HandLandmarker).
No dependency on the legacy `mediapipe.solutions` module and NO contour fallback:
if tracking can't start, we fail loudly instead of feeding bad features to the models.

Self-test:  python hand_tracker.py
"""
import os
import sys
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from config import HAND_MODEL_PATH, HAND_MODEL_URL

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_model(path: str = HAND_MODEL_PATH) -> str:
    """Make sure hand_landmarker.task exists locally; download it if missing."""
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[INFO] Downloading hand_landmarker.task -> {path}")
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, path)
    except Exception as e:
        raise RuntimeError(
            f"Could not download the hand landmarker model ({e}).\n"
            f"Download it manually:\n  curl -L -o {path} {HAND_MODEL_URL}"
        )
    return path


class HandTracker:
    """
    running_mode:
      "image" - independent frames (static datasets)
      "video" - sequential frames with temporal tracking (live webcam, video clips)
    detect() returns a (21, 3) float32 array of normalized image coords, or None.
    """

    def __init__(self, running_mode: str = "video", max_hands: int = 1,
                 min_detection=0.5, min_presence=0.5, min_tracking=0.5):
        ensure_model()
        base = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        mode = (vision.RunningMode.VIDEO if running_mode == "video"
                else vision.RunningMode.IMAGE)
        options = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=mode,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._video = running_mode == "video"
        self._t0 = time.monotonic()
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = None):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(rgb))
        if self._video:
            if timestamp_ms is None:
                timestamp_ms = int((time.monotonic() - self._t0) * 1000)
            if timestamp_ms <= self._last_ts:      # timestamps must strictly increase
                timestamp_ms = self._last_ts + 1
            self._last_ts = timestamp_ms
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None
        return np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]],
                        dtype=np.float32)

    def close(self):
        try:
            self._landmarker.close()
        except Exception:
            pass


def draw_hand(frame: np.ndarray, pts: np.ndarray):
    """Draw the 21-point skeleton on `frame` (pts are normalized coords in this frame's space)."""
    h, w = frame.shape[:2]
    px = [(int(x * w), int(y * h)) for x, y, _ in pts]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], (0, 255, 0), 2)
    for p in px:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def _selftest():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)
    tracker = HandTracker("video")
    print("[OK] HandLandmarker initialized. Show your hand. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        pts = tracker.detect(frame)
        if pts is not None:
            draw_hand(frame, pts)
            msg, color = f"21 landmarks OK (wrist z={pts[0, 2]:+.3f})", (0, 255, 0)
        else:
            msg, color = "No hand", (0, 0, 255)
        cv2.putText(frame, msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("hand_tracker self-test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _selftest()