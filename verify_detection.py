import cv2
from ultralytics import YOLO

print("Loading YOLOv8m...")
model = YOLO('yolov8m.pt')

print("Reading debug_frame.jpg...")
frame = cv2.imread('debug_frame.jpg')
if frame is None:
    print("Error: Could not read debug_frame.jpg")
    exit()

print(f"Frame shape: {frame.shape}")

print("Running inference (CPU)...")
results = model(frame, device='cpu')

for result in results:
    print(f"Detections: {len(result.boxes)}")
    for box in result.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        print(f" - {label}: {conf:.2f}")

print("Done.")
