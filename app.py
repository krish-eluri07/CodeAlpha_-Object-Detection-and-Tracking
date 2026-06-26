import cv2
from ultralytics import YOLO
import supervision as sv

def run_tracking():
    # 1. Initialize YOLO model (using YOLO11n for state-of-the-art performance)
    model = YOLO('yolo11n.pt') 
    
    # 2. Initialize ByteTrack
    tracker = sv.ByteTrack()

    # 3. Initialize Annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # 4. Open Webcam (0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 5. Inference
        # stream=True enables efficient generator-based inference
        results = model(frame, stream=True, verbose=False)

        for result in results:
            # Convert results to supervision Detections
            detections = sv.Detections.from_ultralytics(result)
            
            # Update Tracker
            detections = tracker.update_with_detections(detections)

            # Define custom labels (ID + Class Name)
            labels = [
                f"ID: {tracker_id} {model.names[class_id]}"
                for class_id, tracker_id
                in zip(detections.class_id, detections.tracker_id)
            ]

            # Annotate the frame
            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, 
                detections=detections, 
                labels=labels
            )

            # Display
            cv2.imshow("Real-Time Object Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracking()