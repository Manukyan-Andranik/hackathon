import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
MODEL = YOLO('yolov8n.pt')  # Replace with a custom-trained model if needed

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)

# Define the intersection area (manually set the region of interest)
INTERSECTION_BOX = (400, 400, 400, 400)  # (x1, y1, x2, y2)

def is_blockage(detections, intersection_box):
    """
    Determines if there is a blockage in the intersection area.
    
    Args:
        detections (list): List of detected objects with bounding boxes.
        intersection_box (tuple): The defined intersection box (x1, y1, x2, y2).
        
    Returns:
        bool: True if blockage is detected, otherwise False.
    """
    vehicle_count = 0  # Counter for vehicles in the intersection

    for detection in detections:
        x_min, y_min, x_max, y_max = detection.xyxy[0]  # Bounding box coordinates
        class_id = int(detection.cls[0])  # Class ID
        confidence = detection.conf[0]  # Confidence score

        # Check if the object is within the intersection box
        if (x_min >= intersection_box[0] and y_min >= intersection_box[1] and
            x_max <= intersection_box[2] and y_max <= intersection_box[3]):
            # Count only relevant classes (e.g., cars, trucks, buses)
            if class_id in [2, 5, 7]:  # Class IDs for car, bus, and truck
                vehicle_count += 1
        
        # If 5 or more vehicles are detected, return True
        if vehicle_count >= 3:
            return True

    # Return False if fewer than 5 vehicles are detected
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    
    # Predict on the current frame
    results = MODEL(frame)

    # Annotate the frame with YOLO predictions
    annotated_frame = results[0].plot()

    # Draw the intersection box
    cv2.rectangle(annotated_frame, 
                  (INTERSECTION_BOX[0], INTERSECTION_BOX[1]), 
                  (INTERSECTION_BOX[2], INTERSECTION_BOX[3]), 
                  (0, 255, 0), 2)

    # Check for blockage
    blockage_detected = is_blockage(results[0].boxes, INTERSECTION_BOX)
    
    # Draw a circle indicating blockage status
    if blockage_detected:
        cv2.circle(annotated_frame, (50, 50), 20, (0, 255, 0), -1)  # Green circle
    else:
        cv2.circle(annotated_frame, (50, 50), 20, (0, 0, 255), -1)  # Red circle

    # Display the frame with annotations
    cv2.imshow("YOLOv8 Intersection Monitoring", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
