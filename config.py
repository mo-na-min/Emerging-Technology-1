import os

# Landmark & Tracking Parameters
NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # 63 dimensions (x, y, z per landmark)
SEQUENCE_LENGTH = 30             # Rolling buffer size T = 30 frames

# Motion Arbitration Parameters
# Motion is measured in HAND-SIZE units per frame (wrist->middle-MCP distance = 1.0),
# averaged over the last MOTION_WINDOW frames. Watch the "Motion:" readout in the
# side panel while holding still, then set MOTION_THRESHOLD to ~2.5x that value.
MOTION_THRESHOLD = 0.05          # enter "Motion detected" at/above this
MOTION_EXIT_RATIO = 0.6          # leave motion state below THRESHOLD * RATIO (hysteresis)
MOTION_WINDOW = 5                # frames used for the motion average

CONFIDENCE_THRESHOLD_STATIC = 0.50
CONFIDENCE_THRESHOLD_DYNAMIC = 0.60
SMOOTHING_WINDOW = 12            # Frame window size for majority vote
DWELL_TIME_SECONDS = 1.5         # Hold time needed to commit prediction to text builder

# Camera / mirroring
# Training extraction never flips images. False = the model sees the raw (unflipped)
# camera frame, matching training. The on-screen preview is always mirrored.
# If asymmetric signs are consistently wrong, try True.
MODEL_SEES_MIRRORED = False

# Class Definitions
STATIC_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]

# File Storage Paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models_saved')
STATIC_MODEL_PATH = os.path.join(MODEL_DIR, 'static_rf_model.pkl')
DYNAMIC_MODEL_PATH = os.path.join(MODEL_DIR, 'dynamic_gru_model.pth')
DYNAMIC_VOCAB_PATH = os.path.join(MODEL_DIR, 'dynamic_vocab.json')

# MediaPipe Tasks hand landmarker bundle
HAND_MODEL_PATH = os.path.join(MODEL_DIR, 'hand_landmarker.task')
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

HISTORY_DIR = os.path.join(os.path.dirname(__file__), 'history')
HISTORY_LOG_PATH = os.path.join(HISTORY_DIR, 'history_log.csv')