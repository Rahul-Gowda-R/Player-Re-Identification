from ultralytics import YOLO
import cv2
import os
from utils.tracker import CentroidTracker

model_path = "models/yolov11_weights.pt"
video_path = "data/15sec_input_720p.mp4"

print("[INFO] Checking files...")

if not os.path.exists(model_path):
    print("[ERROR] Model not found at:", model_path)
    exit()

if not os.path.exists(video_path):
    print("[ERROR] Video not found at:", video_path)
    exit()

print("[INFO] Loading model...")
model = YOLO(model_path)
print("[INFO] Model loaded successfully.")

tracker = CentroidTracker()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("[ERROR] Video can't be opened.")
    exit()

print("[INFO] Running detection and tracking...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Done reading frames.")
        break

    results = model(frame)[0]

    player_boxes = []
    for r in results.boxes:
        cls_id = int(r.cls.item())
        if cls_id == 0: 
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            player_boxes.append((x1, y1, x2, y2))
            print(f"Detected player: ({x1}, {y1}, {x2}, {y2})")

    tracked_objects = tracker.update(player_boxes)
    print("Tracked IDs:", tracked_objects)  

    annotated = results.plot()

    for objectID, centroid in tracked_objects.items():
        cv2.putText(annotated, f"Player {objectID}", (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(annotated, (centroid[0], centroid[1]), 5, (0, 255, 0), -1)

    cv2.imshow("Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("done tracking.")
