import os
import csv
import cv2
from config import HISTORY_DIR, HISTORY_LOG_PATH

def show_history():
    """Simple desktop image gallery viewer for logged FSL detections."""
    if not os.path.exists(HISTORY_LOG_PATH):
        print("No history log file found.")
        return

    records = []
    with open(HISTORY_LOG_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    if not records:
        print("History log is empty.")
        return

    print(f"Found {len(records)} history entries. Press 'n' for next, 'p' for previous, 'q' to exit viewer.")
    
    idx = len(records) - 1
    while True:
        item = records[idx]
        img_file = os.path.join(HISTORY_DIR, item['image_filename'])
        if os.path.exists(img_file):
            img = cv2.imread(img_file)
            overlay_text = f"Label: {item['label']} ({item['tier']}) | Conf: {item['confidence']} | {item['timestamp'][:19]}"
            cv2.putText(img, overlay_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("FSL Recognition History Viewer", img)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx = (idx - 1) % len(records)
        elif key == ord('p'):
            idx = (idx + 1) % len(records)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_history()