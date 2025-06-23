import cv2
import numpy as np
import json
from ultralytics import YOLO
import matplotlib.cm as cm # Import colormaps
import matplotlib.colors as mcolors # Import color handling
from collections import deque # Use deque for efficient storage of trail points
import random # Import random for initial color generation

class Detection:
    """A class for tracking object detections and managing tracking visualization."""
    
    def __init__(self, bbox, confidence, class_id):
        """
        Initialize a detection object.
        
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            confidence: Detection confidence score
            class_id: Class ID of the detected object
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.used = False

    def bbox_center(self):
        """Calculate the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def initialize_video_objects(video_path):
        """
        Initialize video capture and get video properties.
        
        Args:
            video_path: Path to the input video file
            
        Returns:
            tuple: (video_capture, width, height, fps)
        """
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        return cap, width, height, fps

    @staticmethod
    def create_video_writer(output_path, width, height, fps):
        """
        Create a video writer object.
        
        Args:
            output_path: Path to save the output video
            width: Frame width
            height: Frame height
            fps: Frames per second
            
        Returns:
            VideoWriter object
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    @staticmethod
    def get_track_color(track_colors, track_id, colormap):
        """
        Get or assign a color for a track ID.
        
        Args:
            track_colors: Dictionary of existing track colors
            track_id: Track ID to get color for
            colormap: Matplotlib colormap to use
            
        Returns:
            BGR color tuple
        """
        if track_id not in track_colors:
            color_index = track_id % 20
            rgb_color = colormap(color_index)[:3]
            bgr_color = (
                int(rgb_color[2] * 255),
                int(rgb_color[1] * 255),
                int(rgb_color[0] * 255)
            )
            track_colors[track_id] = bgr_color
        return track_colors[track_id]

    @staticmethod
    def update_track_history(track_history, track_id, current_center, trail_length=50):
        """
        Update the track history with new center points.
        
        Args:
            track_history: Dictionary of track histories
            track_id: Track ID to update
            current_center: Current center point (x, y)
            trail_length: Maximum length of trail to keep
            
        Returns:
            Updated track_history
        """
        if track_id not in track_history:
            track_history[track_id] = deque(maxlen=trail_length)
        track_history[track_id].append(current_center)
        return track_history

    @staticmethod
    def match_detection_to_track(detections, current_center, max_distance_sq=400):
        """
        Match a track to the closest detection for class information.
        
        Args:
            detections: List of Detection objects
            current_center: Current center point of the track
            max_distance_sq: Maximum allowed squared distance for matching
            
        Returns:
            Matched Detection object or None
        """
        matched_det = None
        min_dist_sq = float('inf')

        for det in detections:
            if not det.used:
                det_center = det.bbox_center()
                dist_sq = (current_center[0] - det_center[0])**2 + (current_center[1] - det_center[1])**2
                if dist_sq < max_distance_sq and dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    matched_det = det
                    matched_det.used = True

        return matched_det

    @staticmethod
    def draw_tracking_elements(frame, tracks, track_history, track_colors, detections, model, colormap):
        """
        Draw all tracking elements on the frame.
        
        Args:
            frame: Current video frame
            tracks: List of active tracks
            track_history: Dictionary of track histories
            track_colors: Dictionary of track colors
            detections: List of current frame detections
            model: Object detection model with class names
            colormap: Color mapping function
            
        Returns:
            tuple: (annotated_frame, current_tracked_ids)
        """
        current_tracked_ids = set()

        for track in tracks:
            if len(track) >= 5:  # Check if track has [x1, y1, x2, y2, id, ...]
                x1, y1, x2, y2, track_id = track[:5]
                track_id = int(track_id)
                current_tracked_ids.add(track_id)

                current_color = Detection.get_track_color(track_colors, track_id, colormap)
                current_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                Detection.update_track_history(track_history, track_id, current_center)

                matched_det = Detection.match_detection_to_track(detections, current_center)

                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), current_color, 2)

                # Draw class name if available
                if matched_det and hasattr(model, 'names'):
                    class_name = model.names[matched_det.class_id]
                    text = f'{class_name}'
                    cv2.putText(frame, text, (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 2)

        # Draw trails
        for track_id, centers in track_history.items():
            if track_id in track_colors and centers:
                trail_color = track_colors[track_id]
                for i in range(1, len(centers)):
                    cv2.line(frame, centers[i-1], centers[i], trail_color, 2)

        # Add tracking count text
        cv2.putText(frame, f"{len(current_tracked_ids)} Birds Tracked", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        return frame, current_tracked_ids
