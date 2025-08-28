from ultralytics import YOLO
import cv2
import math as m

# Pixels → cm conversion factor
distance_threshold = 0.06912

# Load YOLOv8 model (nano version)
model = YOLO("yolov8n.pt")  # Automatically downloads if not present

# Start webcam
cap = cv2.VideoCapture(0)


while True:
    ok, frame = cap.read()
    if not ok:
        print("Couldn't detect frame from camera")
        break

    # Run YOLO inference
    results = model(frame, verbose=False)

    # Store centers of detected objects
    centers = []
    for i, box in enumerate(results[0].boxes):
        if i >= 2:   # kept only 2 detections
            break
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        centers.append((cx, cy))

        # Draw bounding box & label
        label = model.names[int(box.cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        #cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # If at least 2 objects detected, calculate distance
    if len(centers) >= 2:
        x1, y1 = centers[0]
        x2, y2 = centers[1]
        distance = m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * distance_threshold

        # Draw line & distance
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(frame, f"{distance:.2f} cm", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("YOLOv8 Distance Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
