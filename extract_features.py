import cv2
import json
import os
from ultralytics import YOLO
from utils.tracker import CentroidTracker


video_path = "data/broadcast.mp4"  
model_path = "models/yolov11_weights.pt"
output_json = "data/broadcast_tracking.json"

if not os.path.exists(video_path):
    print("❌ video missing:", video_path)
    exit()

if not os.path.exists(model_path):
    print("❌ model not found:", model_path)
    exit()


model = YOLO(model_path)
tracker = CentroidTracker()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ couldn't open video")
    exit()

frame_idx = 0
results = {}

print("[tracking started]")
while True:
    ok, frame = cap.read()
    if not ok: break

    dets = model(frame)[0]
    boxes = []

    for b in dets.boxes:
        cls = int(b.cls.item())
        if cls == 1:  
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            boxes.append((x1, y1, x2, y2))

    tracked = tracker.update(boxes)

    frame_players = []
    for tid, (cx, cy) in tracked.items():
        for bx in boxes:
            if bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3]:
                frame_players.append({
                    "id": int(tid),
                    "bbox": list(map(int, bx)),
                    "centroid": [int(cx), int(cy)]
                })

    results[f"frame_{frame_idx}"] = frame_players
    frame_idx += 1

cap.release()

try:
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[✔] saved to {output_json}")
except Exception as err:
    print("💥 error saving JSON:", err)
