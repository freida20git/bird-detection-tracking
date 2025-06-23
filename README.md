# 🐦 Bird Detection and Tracking Project

This project explores bird detection and multi-object tracking (MOT) using YOLO models and various tracking algorithms. We evaluate and fine-tune models for optimal performance on bird flight videos and provide a complete pipeline to run detection and tracking on any custom video input.

---

## 📁 Project Structure

bird-detection-tracking/ folder structure:
TrackANDcount
inference and results
model
notebooks
track and metric function files

---

## 🧠 Project Overview

###  1. YOLO Detection on Bird Videos
We first tested pre-trained YOLO models (e.g YOLOv8x, YOLOv11n, YOLOv11x) on bird flight videos and observed that larger models perform better in both detection and tracking (Table 1 in `results/results_tables.html`).

We fine-tuned the smallest model yolov11n for bird detection to make it lightweight yet accurate and good for real-time applications.

- **Training Script:** `detection/birds_training_detection.iypnb`  
- **Training Data:**
  data used could be found and downloaded from:
  https://drive.google.com/drive/folders/1k42ZKG3CODvPaURgvAaTu9lUCB-XRrVZ?usp=sharing 
  as well as from Roboflow: 
  https://universe.roboflow.com/sky-sd2zq/bird_only-pt0bm/dataset/1

---

### 🛰️ 2. DeepSORT Tracking Optimization
We started with the DeepSORT tracker and optimized its parameters to work well on our custom bird videos.

- **Tracking Videos:** [Drive Link - Videos](https://drive.google.com/drive/folders/1k6zdJ7NJX8lAgpqBogU346DZoiPj-CO5?usp=drive_link)  
- **Optimized Results:** Table 2 in `results/results_tables.html`

Since no labeled dataset for flying birds was found, we created our own ground truth by:
- Running detections with `yolov11x.pt`
- Manually fixing the results (e.g., ID switches)

- **GT Annotations:** [Drive Link - GT JSONs](https://drive.google.com/drive/folders/1lPxmAk2Akj-ELYQ7_9tSIkyaokVhivvE?usp=drive_link)

---
### 🧪 3. Comparison of Tracking Algorithms
We compared several tracking algorithms on 5 videos (~3,243 image frames in total):

- ✅ SORT  
- ✅ BotSORT  
- ✅ ByteTrack  
- ✅ DeepSORT  
- ✅ OC-SORT  

Each was tested with its **optimal parameters**, and we evaluated their:

- Tracking accuracy (IDF1, MOTA, MOTP, ID_switches)
- Runtime (CPU inference time)

**Results:**
- **OC-SORT** and **DeepSORT** perform best.
- **OC-SORT** offers better inference time while keeping high accuracy.
- **SORT** can be surprisingly good in specific cases.

- **Code for Evaluation:**  
  - `tracking/tracker.ipynb` and `tracking/func_ocsort.ipynb`: run all trackers  
  - `tracking/tuning_tracker_params.ipynb`: parameter optimization  
  - `results/algorithms_inference_results.ipynb`: algorithm comparison (Table 3 in results/results_tables.html)

---

## 🏁 Final Pipeline

We created a user-friendly pipeline to process any `.mp4` video and return:
- Detection + tracking overlay video
- Bird count
- Option to select tracker: `SORT`, `DeepSORT`, or `OC-SORT`

- **Main Code Location:**  
  - Folder: `main_pipeline/`  

---
## 📊 Performance Summary

We show that our fine-tuned model + `OC-SORT` pipeline **outperforms YOLOv11n/m baselines** in detection and tracking accuracy across all 5 videos.  
See **Table 4** in `results/results_tables.html` for full comparison.

---
## 📈 Evaluation Metrics Used

| Metric         | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| **IDF1**       | ID-based F1 score — measures how well identities are preserved.            |
| **MOTA**       | Multi-Object Tracking Accuracy — combines FP, FN, and ID switches.         |
| **MOTP**       | Multi-Object Tracking Precision — how precisely objects are localized.     |
| **mAP@50**     | Mean Average Precision at IoU 0.5 — detection metric.                      |
| **mAP@50:95**  | Mean Average Precision averaged over IoUs 0.5 to 0.95.                     |
| **ID Switches**| Number of identity switches across frames.                                 |
| **CPU Inference Time** | Runtime in seconds per frame.                                |

---
## 🧪 Results Notebooks

- `results/Tracking_metrics.ipynb` – tracking algorithm evaluation  
- `results/Algorithms_inference_results.ipynb` – metric comparison plots  
- `results/model_on_videos.ipynb` – OcSORT model performance per video  

---

## 🧰 Utility Functions

Found in `track_and_metric_func/` folder:
- Reusable functions for metrics computation
- Helpers for running trackers and plotting results
- Used across multiple notebooks

---

## 🚀 How to Run the Project

> *(Instructions to be added by collaborator)*  
To use the detection and tracking pipeline on your own `.mp4` video:

---

## 📎 Links

- 📂 [Bird Videos Dataset](https://drive.google.com/drive/folders/1k6zdJ7NJX8lAgpqBogU346DZoiPj-CO5?usp=drive_link)  
- 📂 [Ground Truth Annotations](https://drive.google.com/drive/folders/1lPxmAk2Akj-ELYQ7_9tSIkyaokVhivvE?usp=drive_link)

---
