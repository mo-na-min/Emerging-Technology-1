"""
Live probe: shows static-model top-3 predictions for the hand as tracked (AS-IS)
and with the x-axis mirrored (MIRRORED), side by side, with no thresholds.

Hold ONE known sign (e.g. 'A') steadily and see which row gives the right letter
with high confidence.

Run:  python live_probe.py
"""
import sys

import cv2
import joblib
import numpy as np

from config import STATIC_MODEL_PATH
from data_loader import normalize_landmarks
from hand_tracker import HandTracker, draw_hand


def top3(clf, vec):
    probs = clf.predict_proba(vec.reshape(1, -1))[0]
    idx = np.argsort(probs)[::-1][:3]
    return [(str(clf.classes_[i]), float(probs[i])) for i in idx]


def mirror_x(vec):
    v = vec.reshape(21, 3).copy()
    v[:, 0] *= -1.0
    return v.flatten()


def draw_rows(frame, title, preds, y0):
    cv2.putText(frame, title, (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i, (label, p) in enumerate(preds):
        color = (0, 255, 0) if p >= 0.5 else (0, 165, 255) if p >= 0.3 else (0, 0, 255)
        cv2.putText(frame, f"{label}: {p * 100:5.1f}%", (24, y0 + 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def main():
    try:
        clf = joblib.load(STATIC_MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Cannot load static model: {e}")
        sys.exit(1)
    if hasattr(clf, "n_jobs"):
        clf.n_jobs = 1
    print(f"[INFO] Model expects {clf.n_features_in_} features; "
          f"{len(clf.classes_)} classes: {list(clf.classes_)}")

    tracker = HandTracker("video")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    print("Hold one known sign. Press 'q' to quit.")
    while True:
        ret, raw = cap.read()
        if not ret:
            break
        display = cv2.flip(raw, 1)
        pts = tracker.detect(raw)               # same frame orientation as training

        if pts is not None:
            draw_pts = pts.copy()
            draw_pts[:, 0] = 1.0 - draw_pts[:, 0]
            draw_hand(display, draw_pts)

            vec = normalize_landmarks(pts)
            draw_rows(display, "AS-IS (MODEL_SEES_MIRRORED=False)", top3(clf, vec), 30)
            draw_rows(display, "X-MIRRORED", top3(clf, mirror_x(vec)), 140)
        else:
            cv2.putText(display, "No hand", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("live_probe", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()