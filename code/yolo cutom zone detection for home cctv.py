import cv2
import math
from ultralytics import YOLO
import numpy as np

# Path to the .mp4 video file
video_path = 0

# Load the model
model = YOLO('/home/ojas-t/yolo/moble/yolov8/home.pt')

# Reading the classes
classnames = ['car', 'biker', 'Person', 'motorbike', 'truck']

# Variables to store polygon points
polygon_points = []
drawing = False

# Mouse callback function to define the polygon detection zone
def set_detection_zone(event, x, y, flags, param):
    global polygon_points, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            polygon_points = [(x, y)]
            drawing = True
        else:
            polygon_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and drawing:
        drawing = False

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', set_detection_zone)

# Open the video file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, stream=True)

    # Draw the detection zone on the frame if defined
    if len(polygon_points) > 1:
        cv2.polylines(frame, [np.array(polygon_points)], isClosed=True, color=(255, 0, 0), thickness=2)
    if len(polygon_points) > 0:
        for point in polygon_points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
        # Display the number of sides
        cv2.putText(frame, f'Sides: {len(polygon_points)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Getting bbox, confidence, and class names information to work with
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Check if the detection is within the detection zone if defined
                if len(polygon_points) > 1:
                    bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    if cv2.pointPolygonTest(np.array(polygon_points), bbox_center, False) >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        class_name = classnames[Class]
                        cv2.putText(frame, f'{class_name}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        # Print the detected class
                        print(f'Detected: {class_name}')

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        polygon_points = []
        drawing = False

cap.release()
cv2.destroyAllWindows()
