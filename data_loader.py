import os
import glob
import numpy as np
import cv2

from hand_tracker import HandTracker

_image_tracker = None


def _get_image_tracker() -> HandTracker:
    global _image_tracker
    if _image_tracker is None:
        _image_tracker = HandTracker("image")
    return _image_tracker


def release_trackers():
    global _image_tracker
    if _image_tracker is not None:
        _image_tracker.close()
        _image_tracker = None


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalizes 3D or 2D landmarks relative to wrist and bounding scale, returning 63-D vector.
    NOTE: this is the normalization the models were trained with; keep it identical at inference."""
    if landmarks is None or len(landmarks) == 0:
        return np.zeros(63, dtype=np.float32)

    landmarks = np.array(landmarks, dtype=np.float32)

    if landmarks.ndim == 2:
        if landmarks.shape[1] == 2:
            landmarks = np.hstack([landmarks, np.zeros((len(landmarks), 1), dtype=np.float32)])
        wrist = landmarks[0]
        centered = landmarks - wrist
        max_val = np.max(np.abs(centered))
        if max_val > 0:
            centered /= max_val
        return centered.flatten().astype(np.float32)
    elif landmarks.ndim == 1 and len(landmarks) == 63:
        pts = landmarks.reshape(21, 3)
        wrist = pts[0]
        centered = pts - wrist
        max_val = np.max(np.abs(centered))
        if max_val > 0:
            centered /= max_val
        return centered.flatten().astype(np.float32)

    return np.zeros(63, dtype=np.float32)


def extract_landmarks_from_image(image_bgr: np.ndarray):
    """Returns normalized 63-D vector from real 21-point tracking, or None if no hand found.
    There is deliberately NO fallback: bad features are worse than no features."""
    if image_bgr is None:
        return None
    pts = _get_image_tracker().detect(image_bgr)
    if pts is None:
        return None
    return normalize_landmarks(pts)


def extract_landmarks_from_video(video_path: str) -> list:
    """Extracts per-frame normalized 63-D vectors from a clip. Frames with no hand are skipped."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = HandTracker("video")
    frames_landmarks = []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pts = tracker.detect(frame, timestamp_ms=int(idx * 1000.0 / fps))
        idx += 1
        if pts is not None:
            frames_landmarks.append(normalize_landmarks(pts))

    tracker.close()
    cap.release()
    return frames_landmarks


def resample_sequence(seq: list, target_seq_len: int = 30) -> np.ndarray:
    """Strictly pads or resamples sequences into uniform numpy shape (30, 63)."""
    if len(seq) == 0:
        return np.zeros((target_seq_len, 63), dtype=np.float32)

    cleaned_seq = []
    for f in seq:
        norm_f = normalize_landmarks(np.array(f))
        if norm_f.shape == (63,):
            cleaned_seq.append(norm_f)
        else:
            cleaned_seq.append(np.zeros(63, dtype=np.float32))

    cleaned_seq = np.array(cleaned_seq, dtype=np.float32)
    curr_len = len(cleaned_seq)

    if curr_len == target_seq_len:
        return cleaned_seq
    elif curr_len < target_seq_len:
        padding = np.tile(cleaned_seq[-1:], (target_seq_len - curr_len, 1))
        return np.vstack([cleaned_seq, padding])
    else:
        indices = np.linspace(0, curr_len - 1, target_seq_len, dtype=int)
        return cleaned_seq[indices]


def load_static_dataset(dataset_path: str):
    """Loads static dataset. Images where no hand is detected are skipped (and counted)."""
    X, y = [], []
    skipped = 0
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Path '{dataset_path}' does not exist.")

    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"[INFO] Processing {len(subdirs)} static classes inside '{dataset_path}'...")

    for label in subdirs:
        class_dir = os.path.join(dataset_path, label)
        kept = 0
        for fname in os.listdir(class_dir):
            fpath = os.path.join(class_dir, fname)
            if fname.endswith('.npy'):
                vec = np.load(fpath)
                norm_v = normalize_landmarks(vec)
                if norm_v.shape == (63,) and np.any(norm_v != 0):
                    X.append(norm_v)
                    y.append(label)
                    kept += 1
                else:
                    skipped += 1
            elif fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                img = cv2.imread(fpath)
                norm_v = extract_landmarks_from_image(img) if img is not None else None
                if norm_v is not None:
                    X.append(norm_v)
                    y.append(label)
                    kept += 1
                else:
                    skipped += 1
        print(f"  -> Static class '{label}': {kept} samples")

    print(f"[INFO] Skipped (no hand found / unreadable): {skipped}")
    release_trackers()
    return np.array(X, dtype=np.float32), np.array(y)


def load_dynamic_dataset(dataset_path: str, target_seq_len: int = 30):
    """Loads dynamic video dataset ensuring uniform (N, 30, 63) output tensor."""
    X_seqs, y_labels = [], []
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Path '{dataset_path}' does not exist.")

    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"[INFO] Processing {len(subdirs)} dynamic classes...")

    for label in subdirs:
        class_dir = os.path.join(dataset_path, label)

        array_files = glob.glob(os.path.join(class_dir, "*.npy"))
        video_files = glob.glob(os.path.join(class_dir, "*.mp4")) + glob.glob(os.path.join(class_dir, "*.avi"))

        loaded = 0
        for spath in array_files:
            try:
                seq = np.load(spath)
                if seq.ndim in [2, 3] and len(seq) > 0:
                    fixed_seq = resample_sequence(seq, target_seq_len)
                    X_seqs.append(fixed_seq)
                    y_labels.append(label)
                    loaded += 1
            except Exception:
                continue

        for vpath in video_files:
            try:
                seq = extract_landmarks_from_video(vpath)
                if len(seq) > 0:
                    fixed_seq = resample_sequence(seq, target_seq_len)
                    X_seqs.append(fixed_seq)
                    y_labels.append(label)
                    loaded += 1
                else:
                    print(f"  [WARN] No hand found in {os.path.basename(vpath)}")
            except Exception as e:
                print(f"  [WARN] Failed on {os.path.basename(vpath)}: {e}")
                continue

        print(f"  -> Dynamic Class '{label}': {loaded} sequences loaded.")

    return np.array(X_seqs, dtype=np.float32), np.array(y_labels)