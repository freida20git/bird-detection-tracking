# bird-detection-tracking
Bird(object) detection in images and tracking in videos using YOLO


📁 Repository Contents:

_Training_:
**birds_training_detection.ipynb**
notebook for training and validating the best-performing model on bird-only data.

_Tracking_:
**Tracker.ipynb**
Tracking implementation using the YOLOv11x model and our custom-trained model.

**sort.py**
Updated python code of sort based on the original with minor modifications to match Google Colab.
Includes tracking from type DeepSORT, SORT, boTSORT and ByteTrack

_Tracking Metrics_:
**Tracking_metrics.ipynb**
Evaluation metrics and performance analysis for the tracking results.

_Hyperparameter Tuning_:
**tuning_tracker_params.ipynb**
Experiments with 10 different hyperparameter configurations and all option combinations to identify optimal settings.

**DATA:**
data used could be found and downloaded from:

https://drive.google.com/drive/folders/1k42ZKG3CODvPaURgvAaTu9lUCB-XRrVZ?usp=sharing 

as well as from Roboflow: 

https://universe.roboflow.com/sky-sd2zq/bird_only-pt0bm/dataset/1
