from sort import Sort, KalmanBoxTracker
from VideoDrawings import *

def initialize_sort_tracker():
    """Initialize and return the SORT tracker."""
    KalmanBoxTracker.count = 0
    return Sort(max_age=30, min_hits=3, iou_threshold=0.1)

def process_sort_detections(results):
    """Process YOLO detection results for SORT using Detection objects."""
    detection_objects = []
    detections_array = []
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        
        # Create Detection object
        det_obj = Detection(
            bbox=(x1, y1, x2, y2),
            confidence=conf,
            class_id=cls_id
        )
        
        detection_objects.append(det_obj)
        detections_array.append([x1, y1, x2, y2, conf])
    
    return np.array(detections_array) if detections_array else np.empty((0, 5)), detection_objects

def main_sort(input_path, output_path, model_name):
    """Main function for SORT tracking."""
    cap, width, height, fps = Detection.initialize_video_objects(input_path)
    out = Detection.create_video_writer(output_path, width, height, fps)
    model = YOLO(model_name)
    tracker = initialize_sort_tracker()

    track_history = {}
    track_colors = {}
    colormap = cm.get_cmap('tab10', 20)
    unique_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.3, classes=0, iou=0.4, verbose=False)
        dets, detections = process_sort_detections(results)
        tracks = tracker.update(dets)

        frame, current_ids = Detection.draw_tracking_elements(
            frame=frame,
            tracks=tracks,
            track_history=track_history,
            track_colors=track_colors,
            detections=detections,
            model=model,
            colormap=colormap
        )
        
        unique_ids.update(current_ids)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Total number of birds detected: {len(unique_ids)}")
