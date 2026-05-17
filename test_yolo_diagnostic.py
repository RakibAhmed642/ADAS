
import os
import cv2
import numpy as np
from ultralytics import YOLO

print("🔍 DIAGNOSTIC: Checking YOLO Model Integrity...")

model_path = 'yolov8m.pt'

if not os.path.exists(model_path):
    print(f"❌ Error: {model_path} not found!")
    exit()

size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"📂 Model Size: {size_mb:.2f} MB")

if size_mb < 1:
    print("❌ Error: Model file is too small! Likely corrupt.")
    exit()

try:
    print("⏳ Loading model...")
    model = YOLO(model_path)
    print("✅ Model loaded successfully.")
    
    print("🧪 Running inference on dummy image...")
    dummy_img = np.zeros((360, 640, 3), dtype=np.uint8)
    # Draw a white rectangle to simulate an object (though model won't detect it, it checks validity)
    cv2.rectangle(dummy_img, (100, 100), (200, 200), (255, 255, 255), -1)
    
    results = model(dummy_img, verbose=True)
    print(f"✅ Inference successful! Detected: {len(results[0].boxes)} objects (Expected 0 or random)")
    
except Exception as e:
    print(f"❌ CRITICAL ERROR: {e}")
    print("👉 RECOMMENDATION: Delete yolov8m.pt and let new.py re-download it.")
