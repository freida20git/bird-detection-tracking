# 🐦 Bird Detection and Tracking Project

This project explores bird detection and multi-object tracking (MOT) using YOLO models and various tracking algorithms. We evaluate and fine-tune models for optimal performance on bird flight videos and provide a complete pipeline to run detection and tracking on any custom video input.

---

## 🚀 How to Run the Project

detailed notebook is in main_pipeline folder and it's a one line code:

> main_ocsort(input_video_path, ocsort_output_path, model_pt_path)

or

>  main_sort(input_video_path, sort_output_path, model_pt_path)

Example output:
![image](https://github.com/user-attachments/assets/00cbdd1f-21a8-4c37-a38d-ecd121e940fb)

---

## 📁 Project Structure

This repository is organized as follows:
Each directory serves a specific role:
- `detection/`: Contains YOLOv11n training notebooks and model.
- `tracking/`: Exploration and evaluation of tracking algorithms.
- `main_pipeline/`: Scripts for combining detection and tracking into a unified user-friendly pipeline.
- `track_and_metric_func/`: Helper functions used in pipelines and metric calculations.
- `results/`: Outputs, plots, and metrics summaries from experiments.
- `requirements.txt`: Python dependencies for the full pipeline.

---

## Project Overview

###  1. YOLO Detection on Bird Videos
We first tested pre-trained YOLO models (e.g YOLOv8x, YOLOv11n, YOLOv11x) on bird flight videos and observed that larger models perform better in both detection and tracking (Table 1 in `results/results_tables.pdf`).

We fine-tuned the smallest model yolov11n for bird detection to make it lightweight yet accurate and good for real-time applications.

- **Training Script:** `detection/birds_training_detection.iypnb`  
- **Training Data:**
  data used could be found and downloaded from: [Drive Link - model_data](https://drive.google.com/drive/folders/1k42ZKG3CODvPaURgvAaTu9lUCB-XRrVZ?usp=sharing). 
  as well as from Roboflow: [Drive Link - roboflow_original_data](https://universe.roboflow.com/sky-sd2zq/bird_only-pt0bm/dataset/1).

---

### 🛰️ 2. DeepSORT Tracking Optimization
We started with the DeepSORT tracker and optimized its parameters to work well on our custom bird videos.

- **Tracking Videos:** [Drive Link - Videos](https://drive.google.com/drive/folders/1k6zdJ7NJX8lAgpqBogU346DZoiPj-CO5?usp=drive_link)  
- **Optimized Results:** Table 2 in `results/results_tables.pdf`

Since no labeled dataset for flying birds was found, we created our own ground truth by:
- Running detections with `yolov11x.pt`
- Manually fixing the results (e.g., ID switches)
  for example we erased multi-objects False Negatives detections like id 20 object in the following frame:
  ![Screenshot (786)](https://github.com/user-attachments/assets/4773225d-9e29-4f28-b426-e9d5f01506e7)


- **GT Annotations:** [Drive Link - GT JSONs](https://drive.google.com/drive/folders/1lPxmAk2Akj-ELYQ7_9tSIkyaokVhivvE?usp=drive_link)

  📄 Annotation JSON Format:
    The annotation file is a list of dictionaries, each representing a video frame with the following structure:
    - **`frame_number`**: Integer indicating the frame index.
    - **`objects`**: A list of detected/tracked objects in the frame, where each object includes:
      - **`track_id`**: Unique identifier for the tracked object.
      - **`class_id`**: Numeric class label.
      - **`class_name`**: Human-readable class name (e.g., `"bird"`).
      - **`confidence`**: Detection confidence score.
      - **`bbox`**: Bounding box coordinates with:
        - `x1`, `y1`: Top-left corner
        - `x2`, `y2`: Bottom-right corner
    Frames without any objects have an empty `objects` list.
    
    📄 for further analysis we transform the annotations to .txt MOT format in the Tracking metrics file which has the following format:
  
     Each detection includes:
  
        frame, track_id, x_center, y_center, width, height, confidence, class_id, visibility.

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

  for example:
  ![image](https://github.com/user-attachments/assets/777e5568-6880-4c85-9c71-8f11065aa014)
  more plots can be seen in `results/algorithms_inference_results.ipynb`.


**Results:**
- **OC-SORT** and **DeepSORT** perform best.
- **OC-SORT** offers better inference time while keeping high accuracy.
- **SORT** can be surprisingly good in specific cases.

- **Code for Evaluation:**  
  - `tracking/tracker.ipynb` and `tracking/func_ocsort.ipynb`: run all trackers  
  - `tracking/tuning_tracker_params.ipynb`: parameter optimization  
  - `results/Algorithms_inference_results.ipynb`: algorithm comparison plots (Table 3 in results/results_tables.pdf)

---

## 🏁 Final Pipeline

We created a user-friendly pipeline to process any `.mp4` video and return:
- Detection + tracking overlay video
- Bird count
- Option to select tracker: `SORT` or `OC-SORT`

- **Main Code Location:**  
  - Folder: `main_pipeline/UsageExample.ipynb`  

---
## 📊 Performance Summary

We show that our fine-tuned model + `OC-SORT` pipeline **outperforms YOLOv11n/m baselines** in detection and tracking accuracy across all 5 videos.  
See **Table 4** in `results/results_tables.pdf` for full comparison.

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


for a deeper theoretical understanding of tracking metrics check: [MOT metrics](https://miguel-mendez-ai.com/2024/08/25/mot-tracking-metrics).

---
## 🧪 Results Notebooks

- `results/Tracking_metrics.ipynb` – tracking algorithm evaluation  
- `results/Algorithms_inference_results.ipynb` – metric comparison plots  
- `results/model_on_videos.ipynb` – OcSORT model performance per video
- `results/results_tables.pdf` - all results summarized in tables.

---

## 🧰 Utility Functions

Found in `track_and_metric_func/` folder:
- Reusable functions for metrics computation
- Helpers for running trackers and plotting results
- Used across multiple notebooks

---

## 📎 Links

- 📂 [Bird Videos Dataset](https://drive.google.com/drive/folders/1k6zdJ7NJX8lAgpqBogU346DZoiPj-CO5?usp=drive_link)  
- 📂 [Ground Truth Annotations](https://drive.google.com/drive/folders/1lPxmAk2Akj-ELYQ7_9tSIkyaokVhivvE?usp=drive_link)
- 📂 [Data used for fine-tuning model](https://drive.google.com/drive/folders/1k42ZKG3CODvPaURgvAaTu9lUCB-XRrVZ?usp=sharing)
 
---

feedback and collaboration ideas are welcome.
