# ⚽ Player Re-Identification in Sports Broadcasts using YOLOv11

This project implements an **AI-powered Player Re-Identification system** in sports videos using the **YOLOv11** object detection model and a **centroid-based tracking** mechanism. It identifies and tracks players across frames, generating persistent IDs for re-identification — useful for analytics, coaching, or post-match breakdowns.

---

## 📌 Problem Statement

In sports video footage, accurately tracking individual players is difficult due to:

- Similar jersey colors
- Fast camera movements and zoom
- Occlusions (players blocking each other)
- Rapid direction changes

This project solves the **single-camera tracking** part of the problem. It uses real-time object detection (YOLOv11) + tracking (CentroidTracker) to assign **unique IDs to players across frames** and saves tracking data for downstream re-identification or analytics.

---

## 🎯 Objectives

- ✅ Detect **players**, **goalkeepers**, and **referees** using a trained YOLOv11 model  
- ✅ Track player movement across video frames using **centroid-based tracking**
- ✅ Assign and display **persistent Player IDs**
- ✅ Save tracking metadata to a `.json` file for further processing
- ✅ Set the foundation for cross-camera re-identification in the future

---

## 🛠️ Tech Stack

| Component      | Tool/Library                |
|----------------|-----------------------------|
| Language       | Python 3.12                 |
| Detection      | [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) |
| Tracking       | Custom Centroid Tracker     |
| Video Handling | OpenCV                      |
| Model Format   | `.pt` (PyTorch checkpoint)  |
| Output Format  | `.json`, annotated video    |

---

## 📁 Project Structure

```

player-reid-liatai/
│
├── data/                          # Input and tracking videos
│   ├── 15sec\_input\_720p.mp4
│   ├── broadcast.mp4
│   └── tacticam.mp4
│
├── models/                        # YOLOv11 weights (not uploaded to GitHub)
│   └── yolov11\_weights.pt
│
├── outputs/                       # JSON & annotated outputs
│   ├── tracked\_video.mp4
│   ├── broadcast\_tracking.json
│   ├── tacticam\_tracking.json
│   └── matched\_ids.json
│
├── detect.py                      # Player detection + tracking
├── extract\_features.py            # (Optional) Feature vector extractor
├── match\_players.py               # Match player IDs across feeds
├── reid.py                        # (Optional) Re-identification logic
├── tracker.py                     # Centroid-based tracker class
├── requirements.txt               # Python dependencies
├── README.md
└── .gitignore

````

---

## ▶️ How to Run

1. **Install dependencies**

```bash
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
````

2. **Download YOLOv11 weights**

> ⚠️ The model weights are not included in this repo due to GitHub's 100MB limit.
> Place the trained `.pt` file as:

```
models/yolov11_weights.pt
```

3. **Run player detection and tracking**

```bash
python detect.py
```

This will:

* Open the input video
* Run detection per frame
* Assign IDs to players
* Annotate the video and display in real time
* Save the output (optional)

---

## 📦 Output

* 🧠 **Live annotated video** with bounding boxes and Player IDs
* 🧾 **JSON files** containing player ID, bounding boxes, and frame numbers
* 🎞️ (Optional) Annotated video file in `outputs/`

---

## 💡 Future Scope

* Add feature-based re-identification using embeddings (via `extract_features.py` and `reid.py`)
* Support for multi-camera player mapping (cross-view matching)
* Use deep SORT or ByteTrack for improved tracking accuracy
* Add player analytics (heatmaps, distance covered, etc.)

---

## 📜 Acknowledgements

* [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
* [Adrian Rosebrock's tracking tutorials](https://www.pyimagesearch.com/)
* Research papers on person re-identification and multi-object tracking

---

## 🧠 Author

**Rahul Gowda R**
Final Year Engineering Student
GitHub: [@Rahul-Gowda-R](https://github.com/Rahul-Gowda-R)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```


