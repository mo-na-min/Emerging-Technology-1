from collections import Counter, deque
import time
import numpy as np
from config import (
    MOTION_THRESHOLD, MOTION_EXIT_RATIO, MOTION_WINDOW,
    SMOOTHING_WINDOW, DWELL_TIME_SECONDS,
    CONFIDENCE_THRESHOLD_STATIC, CONFIDENCE_THRESHOLD_DYNAMIC,
)

NON_COMMIT_LABELS = {"Low confidence", "No hand detected", "Analyzing...", "No prediction", "..."}


class ArbitrationEngine:
    """
    Motion estimation, static-vs-dynamic gating, prediction smoothing, and dwell-to-commit.

    Motion is computed from RAW landmarks scaled by hand size (wrist -> middle MCP = 1.0),
    so whole-hand translation counts as motion (wrist-centered features cannot see it),
    and the threshold is independent of how far the hand is from the camera.
    """

    def __init__(self,
                 static_threshold=CONFIDENCE_THRESHOLD_STATIC,
                 dynamic_threshold=CONFIDENCE_THRESHOLD_DYNAMIC):
        self.static_threshold = static_threshold
        self.dynamic_threshold = dynamic_threshold

        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
        self.raw_buffer = deque(maxlen=MOTION_WINDOW + 1)
        self.in_motion = False
        self.last_motion = 0.0

        self.current_dwell_label = None
        self.dwell_start_time = None
        self.committed_text_builder = []

    def reset(self):
        """Call when the hand is lost. Does NOT clear committed text."""
        self.prediction_history.clear()
        self.raw_buffer.clear()
        self.in_motion = False
        self.last_motion = 0.0
        self.current_dwell_label = None
        self.dwell_start_time = None

    def update_motion(self, raw_pts: np.ndarray) -> float:
        """raw_pts: (21, 3) un-normalized landmarks for the current frame."""
        pts = np.asarray(raw_pts, dtype=np.float32)[:, :2]
        size = float(np.linalg.norm(pts[9] - pts[0]))
        size = max(size, 1e-4)
        self.raw_buffer.append(pts / size)

        if len(self.raw_buffer) < 2:
            motion = 0.0
        else:
            seq = np.stack(self.raw_buffer)                       # (W, 21, 2)
            diffs = np.linalg.norm(seq[1:] - seq[:-1], axis=2)    # (W-1, 21)
            motion = float(np.mean(diffs))

        self.last_motion = motion
        if self.in_motion:
            self.in_motion = motion >= MOTION_THRESHOLD * MOTION_EXIT_RATIO
        else:
            self.in_motion = motion >= MOTION_THRESHOLD
        return motion

    def arbitrate_and_smooth(self, static_pred, dynamic_pred, allow_dynamic=True):
        """
        Returns (smoothed_label, tag, confidence, motion_state).
        If allow_dynamic is False (no GRU loaded) everything is routed to the static tier.
        """
        use_dynamic = self.in_motion and allow_dynamic

        if use_dynamic:
            motion_state = "Motion detected"
            if dynamic_pred is not None:
                raw_label, conf = dynamic_pred
                if conf >= self.dynamic_threshold:
                    candidate = (raw_label, "[WORD]", conf)
                else:
                    candidate = ("Low confidence", "[WORD]", conf)
            else:
                candidate = ("Analyzing...", "[WORD]", 0.0)
        else:
            motion_state = "Static hold" if allow_dynamic else "Static only"
            if static_pred is not None:
                raw_label, conf = static_pred
                raw_label = str(raw_label)
                tag = "[NUMBER]" if raw_label.isdigit() else "[LETTER]"
                if conf >= self.static_threshold:
                    candidate = (raw_label, tag, conf)
                else:
                    candidate = ("Low confidence", tag, conf)
            else:
                candidate = ("No prediction", "[N/A]", 0.0)

        self.prediction_history.append(candidate)

        # Majority vote over labels; tag/confidence come from the winning label's entries
        labels = [c[0] for c in self.prediction_history]
        smoothed_label = Counter(labels).most_common(1)[0][0]
        winners = [c for c in self.prediction_history if c[0] == smoothed_label]
        tag = winners[-1][1]
        conf = float(np.mean([c[2] for c in winners]))

        return smoothed_label, tag, conf, motion_state

    def process_dwell(self, label: str, tag: str, confidence: float) -> bool:
        """Returns True if the held sign was committed to the text builder."""
        if label in NON_COMMIT_LABELS:
            self.current_dwell_label = None
            self.dwell_start_time = None
            return False

        now = time.time()
        if self.current_dwell_label != label:
            self.current_dwell_label = label
            self.dwell_start_time = now
            return False

        if self.dwell_start_time and (now - self.dwell_start_time >= DWELL_TIME_SECONDS):
            if label == "SPACE":
                self.committed_text_builder.append(" ")
            elif label == "BACKSPACE":
                if self.committed_text_builder:
                    self.committed_text_builder.pop()
            elif tag == "[WORD]":
                self.committed_text_builder.append(label + " ")
            else:
                self.committed_text_builder.append(label)

            # Require a longer hold before the same sign can repeat
            self.dwell_start_time = time.time() + 1.0
            return True

        return False