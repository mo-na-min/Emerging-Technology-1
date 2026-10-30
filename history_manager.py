import os
import csv
from datetime import datetime
import cv2

class HistoryLogger:
    """Logs committed sign detections with webcam snapshot frames and metadata."""
    def __init__(self, history_dir: str, log_csv_path: str):
        self.history_dir = history_dir
        self.log_csv_path = log_csv_path
        
        os.makedirs(self.history_dir, exist_ok=True)
        if not os.path.exists(self.log_csv_path):
            with open(self.log_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'label', 'tier', 'confidence', 'image_filename'])

    def log_event(self, frame, label: str, tier: str, confidence: float):
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_name = f"snap_{timestamp_str}.jpg"
        img_path = os.path.join(self.history_dir, img_name)
        
        # Save frame image
        cv2.imwrite(img_path, frame)
        
        # Save log entry
        with open(self.log_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), label, tier, f"{confidence:.2f}", img_name])