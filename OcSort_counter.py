from trackers.ocsort_tracker.ocsort import OCSort
from VideoDrawings import *
def initialize_ocsort_tracker():
    """Initialize and return the OC-SORT tracker."""
    return OCSort(
        det_thresh=0.4,
        max_age=90,
        min_hits=2,
        iou_threshold=0.3,
        delta_t=3,
        inertia=0.3,
        asso_func="ciou"
    )

def process_ocsort_detections(results):
    """Process YOLO detection results for OC-SORT with class IDs."""
    detections = results[0].boxes
    if detections is not None and len(detections) > 0:
        boxes = detections.xyxy.cpu().numpy()
        scores = detections.conf.cpu().numpy()
        class_ids = detections.cls.cpu().numpy()
        dets = np.hstack((boxes, scores.reshape(-1, 1)))

        # Create list of Detection objects instead of dictionaries
        detection_objects = [
            Detection(
                bbox=dets[i][:4],
                confidence=float(dets[i][4]),
                class_id=int(class_ids[i])
            ) for i in range(len(dets))
        ]
        return dets, detection_objects
    return np.empty((0, 5)), []

def main_ocsort(input_path, output_path, model_name):
    """Main function for OC-SORT tracking."""
    # Use the Detection class methods for video initialization
    cap, width, height, fps = Detection.initialize_video_objects(input_path)
    out = Detection.create_video_writer(output_path, width, height, fps)

    model = YOLO(model_name)
    tracker = initialize_ocsort_tracker()

    track_history = {}
    track_colors = {}
    colormap = cm.get_cmap('tab10', 20)
    unique_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.3, iou=0.4, verbose=False)
        dets, detections = process_ocsort_detections(results)
        tracks = tracker.update(dets, frame.shape[:2], frame.shape[:2])

        # Use the Detection class method for drawing
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
