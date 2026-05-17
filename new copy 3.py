# ==========================================
# REAL-TIME ADAS SYSTEM V34.3 (BUG FIX - LABEL ERROR)
# Features: Live Traffic Stats, Red Strip Blind Spot, Traffic Light Filter
# Author: Gemini
# ==========================================

import os
import cv2
import numpy as np
import base64
import time
import torch
import sys
import math
from collections import deque
from datetime import datetime
import threading
from queue import Queue

# --- ADVANCED PERFORMANCE: THREADED VIDEO STREAM ---
class ThreadedVideoStream:
    def __init__(self, src=0, queue_size=3):
        self.stream = cv2.VideoCapture(src)
        if isinstance(src, int):
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.stopped = False
        self.Q = deque(maxlen=queue_size)
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        
    def start(self):
        self.thread.start()
        return self
        
    def update(self):
        while True:
            if self.stopped:
                return
            ret, frame = self.stream.read()
            if not ret:
                self.stopped = True
                return
            self.Q.append(frame)
            
    def read(self):
        if len(self.Q) > 0:
            return True, self.Q[-1]
        return False, None
        
    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.stream.release()



# 1. SETUP & STRICT GPU CHECK
# -----------------------------------
VIDEO_SOURCE = 1  # Set to 0 for Webcam, or path to video file (e.g., "video.mp4")

try:
    from ultralytics import YOLO
except ImportError:
    print("Installing Ultralytics (YOLOv8)... Please wait.")
    os.system('pip install ultralytics')
    from ultralytics import YOLO

# FORCE GPU CHECK
if not torch.cuda.is_available():
    print("❌ ERROR: GPU IS NOT DETECTED!")
    print("Running on CPU. Performance may be slow.")
    print("="*50 + "\n")
    device = 'cpu'
else:
    print(f"✅ SUCCESS: GPU DETECTED -> {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = 'cuda'

print("Loading EFFICIENT & POWERFUL Model (YOLO11m) for Real-Time Accuracy...")
# Using the highly efficient YOLO11m model as requested to avoid crashes and run smoothly
model = YOLO('yolo11x.pt')
if device == 'cuda':
    model.to('cuda')

    # GPU WARMUP
    print("🔥 Warming up GPU...")
    dummy_input = np.zeros((360, 640, 3), dtype=np.uint8)
    for _ in range(3):
        model.predict(dummy_input, verbose=False, half=True)
    print("🚀 GPU Ready!")

# 2. HELPER FUNCTIONS
# -----------------------------------
# (JavaScript helpers removed for local execution)


# --- HELPER: BOX SMOOTHER (EMA) ---
class BoxSmoother:
    def __init__(self, alpha=0.6):
        self.boxes = {}
        self.alpha = alpha

    def update(self, track_id, current_box):
        if track_id not in self.boxes:
            self.boxes[track_id] = current_box
            return current_box

        prev_box = self.boxes[track_id]
        smooth_box = [
            int(self.alpha * curr + (1 - self.alpha) * prev)
            for curr, prev in zip(current_box, prev_box)
        ]
        self.boxes[track_id] = smooth_box
        return smooth_box

# --- ADVANCED MATH: 3-STATE KALMAN FILTER ---
class AdvancedKalmanFilter:
    def __init__(self, dt=1/30):
        self.x = np.zeros((3, 1))
        self.F = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
        self.H = np.array([[1, 0, 0]])
        self.P = np.eye(3) * 100
        self.R = np.array([[5]])
        self.Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 5]])

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[0][0]

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        try:
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if S is singular
            K = np.dot(np.dot(self.P, self.H.T), np.linalg.pinv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(3)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.x[0][0], self.x[1][0], self.x[2][0]

# 3. INTELLIGENT ADAS PROCESSOR
# -----------------------------------
class ADASProcessor:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.inference_time = 0
        self.prev_frame_time = time.time()

        # Histories
        self.kalman_filters = {}
        self.object_paths = {}
        self.last_seen = {}
        self.track_lifetime = {}
        self.lateral_history = {}
        self.advanced_physics_history = {}
        self.smoother = BoxSmoother(alpha=0.5)

        # Lane History
        self.lane_history = deque(maxlen=20)

        # Statistics
        self.stats = { 'car': 0, 'person': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0 }
        self.tracked_ids = set()

        # Signal Tracking History
        # Format: { track_id: [ (timestamp, 'LEFT'/'RIGHT'/'BRAKE'/'OFF'), ... ] }
        self.signal_history = {}
        
        # Directional Tracking (Is vehicle oncoming or same direction?)
        self.direction_history = {}

        # Environment
        self.visibility = "CHECK..."
        self.weather_status = "CLEAR"
        self.road_status = "SMOOTH"
        self.traffic_status = "FREE"
        self.beam_suggestion = "AUTO"
        self.friction_coef = 0.7

        # Camera
        self.cam_height = 1.5
        self.focal_length = 800

        # Tools
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.pothole_alert = False
        self.ldw_warning = None

        # Colors (Expanded with extra vehicles and objects)
        self.colors = {
            'car': (255, 255, 0), 'person': (0, 165, 255),
            'truck': (255, 0, 255), 'bus': (0, 128, 255),
            'motorcycle': (50, 205, 50), 'bicycle': (255, 191, 0),
            'train': (0, 100, 255), 'airplane': (255, 255, 255),
            'boat': (255, 50, 50), 'bird': (100, 255, 100),
            'traffic light': (200, 200, 200), 'stop sign': (0, 0, 200),
            'dog': (180, 105, 255), 'cow': (180, 105, 255),
            'horse': (180, 105, 255), 'sheep': (180, 105, 255),
            'cat': (180, 105, 255), 'bear': (180, 105, 255),
            'parking meter': (200, 200, 50), 'bench': (150, 150, 150),
            'backpack': (0, 255, 255), 'umbrella': (0, 255, 255),
            'handbag': (0, 255, 255), 'suitcase': (0, 255, 255),
            'bottle': (0, 255, 255), 'sports ball': (0, 255, 255),
            'chair': (0, 255, 255), 'couch': (0, 255, 255),
            'potted plant': (0, 255, 255), 'bed': (0, 255, 255)
        }

    def draw_text_with_outline(self, img, text, pos, font_scale, color, thickness=1):
        x, y = pos
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def analyze_environment(self, frame, vehicle_count):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Sky Analysis (Top 30% of frame)
        h, w = frame.shape[:2]
        sky_roi = frame[:int(h*0.3), :]
        sky_hsv = cv2.cvtColor(sky_roi, cv2.COLOR_BGR2HSV)
        
        # Calculate dominant color in sky
        avg_h = np.mean(sky_hsv[:, :, 0])
        avg_s = np.mean(sky_hsv[:, :, 1])
        avg_v = np.mean(sky_hsv[:, :, 2])
        
        # Weather Logic
        # Sharpness is primary for Fog/Rain
        # Color is primary for Sunny/Cloudy
        
        if sharpness < 60:
            self.weather_status = "FOGGY"
            self.friction_coef = 0.5
            self.visibility = "POOR"
        elif sharpness < 120:
            self.weather_status = "RAIN"
            self.friction_coef = 0.4
            self.visibility = "REDUCED"
        else:
            # Check for Blue Sky (Hue ~90-130 in OpenCV Scale?) 
            # Actually standard Blue is ~100-120. 
            # Sunny usually means high brightness and reasonable saturation of blue
            if avg_v > 150: 
                if avg_s > 30: # Saturation implies color (Blue sky)
                    self.weather_status = "SUNNY"
                else:
                    self.weather_status = "CLOUDY" # Bright but gray
            else:
                 self.weather_status = "CLOUDY"  # Darker

            self.friction_coef = 0.8
            self.visibility = "GOOD"

        if vehicle_count > 8: self.traffic_status = "JAM"
        elif vehicle_count > 3: self.traffic_status = "BUSY"
        else: self.traffic_status = "FREE"

        road_roi = gray[int(h*0.7):h, int(w*0.3):int(w*0.7)]
        self.pothole_alert = False
        if road_roi.size > 0:
            mean, std_dev = cv2.meanStdDev(road_roi)
            if std_dev[0][0] > 75: 
                self.road_status = "SEVERE ROAD"
                self.pothole_alert = True
            elif std_dev[0][0] > 50: self.road_status = "BUMPY"
            else: self.road_status = "SMOOTH"

    def apply_night_vision(self, frame, boxes):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        brightness = np.mean(hsv[:, :, 2])
        mode_text = ""

        if brightness < 30: self.visibility = "DARK"
        elif brightness < 70: self.visibility = "DIM"
        elif brightness > 220: self.visibility = "GLARE"
        else: self.visibility = "GOOD"

        if brightness < 70:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = self.clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            mode_text = "NIGHT MODE"

            car_ahead = len(boxes) > 0
            if car_ahead: self.beam_suggestion = "BEAM: LOW"
            else: self.beam_suggestion = "BEAM: HIGH"
        else:
            self.beam_suggestion = "DAY"

        return frame, mode_text

    def estimate_distance_advanced(self, y_bottom, h, w_box, label):
        if y_bottom >= h: return 0.0
        horizon = 0.5 * h
        dist_y = 150.0
        if y_bottom > horizon:
            dy = y_bottom - horizon
            alpha = np.arctan(dy / self.focal_length)
            dist_y = max(0.1, self.cam_height / np.tan(alpha))
            
        dist_w = dist_y
        widths = {'car': 1.8, 'truck': 2.5, 'bus': 2.8, 'motorcycle': 0.8, 'person': 0.5, 'bicycle': 0.6, 'train': 3.0}
        if label in widths and w_box > 0:
            real_w = widths[label]
            dist_w = (real_w * self.focal_length) / w_box
            
        # Blend optical expansion width distance with geometric geometric distance for ultimate accuracy
        if y_bottom > horizon + 20: 
            return (dist_y * 0.6) + (dist_w * 0.4)
        return dist_w

    def update_kalman(self, track_id, measured_dist, dt):
        if track_id not in self.kalman_filters:
            self.kalman_filters[track_id] = AdvancedKalmanFilter(dt)
            self.kalman_filters[track_id].x[0][0] = measured_dist
            self.track_lifetime[track_id] = 0

        kf = self.kalman_filters[track_id]
        kf.F[0, 1] = dt
        kf.F[0, 2] = 0.5 * dt**2
        kf.F[1, 2] = dt
        kf.predict()
        dist, velocity, accel = kf.update(measured_dist)
        self.track_lifetime[track_id] += 1
        return dist, velocity, accel

    def update_trajectory(self, track_id, center_point):
        if track_id not in self.object_paths:
            self.object_paths[track_id] = deque(maxlen=20)
        self.object_paths[track_id].append(center_point)

    def detect_lateral_intent(self, track_id, current_x, dt, width):
        is_cut_in = False
        intent = "STRAIGHT"
        center_screen = width / 2
        if track_id not in self.lateral_history:
            self.lateral_history[track_id] = {'x': current_x, 'v': 0}
            return False, 0, intent
        prev = self.lateral_history[track_id]
        dx = current_x - prev['x']
        lat_velocity = dx / dt
        
        # Smooth lateral velocity for stability
        smooth_lat = (prev['v'] * 0.6) + (lat_velocity * 0.4)
        self.lateral_history[track_id] = {'x': current_x, 'v': smooth_lat}
        
        if smooth_lat < -25: intent = "SHIFT LEFT"
        elif smooth_lat > 25: intent = "SHIFT RIGHT"
        
        moving_to_center = False
        if current_x < center_screen and smooth_lat > 15: moving_to_center = True
        if current_x > center_screen and smooth_lat < -15: moving_to_center = True
        if moving_to_center and abs(smooth_lat) > 40: is_cut_in = True
        
        return is_cut_in, smooth_lat, intent

    def calculate_physics_metrics(self, velocity_mps, distance_m, accel_mps2):
        ttc = float('inf')
        headway = float('inf')
        is_braking = False
        closing_speed = -velocity_mps

        if closing_speed > 0.1: ttc = distance_m / closing_speed
        if closing_speed > 1: headway = distance_m / closing_speed
        if accel_mps2 < -2.0: is_braking = True
        
        # Advanced Math: Impact Severity based on Kinetic Energy absorption estimate
        impact_severity = 0.0
        if ttc < 5.0 and closing_speed > 0:
            impact_severity = 0.5 * closing_speed**2 / 100
            
        return ttc, headway, is_braking, impact_severity

    def calculate_collision_risk(self, ttc):
        if ttc == float('inf'): return 0
        sigma = 3.0
        risk = np.exp(- (ttc**2) / (2 * sigma**2))
        return int(risk * 100)

    def calculate_stopping_distance(self, speed_kmh):
        v_ms = abs(speed_kmh) / 3.6
        reaction_dist = 1.5 * v_ms
        braking_dist = (v_ms**2) / (2 * self.friction_coef * 9.8)
        return reaction_dist + braking_dist

    def calculate_safety_score(self, max_risk_prob, weather, road):
        score = 100
        score -= max_risk_prob
        if "RAIN" in weather: score -= 20
        if "BUMPY" in road: score -= 10
        if "DARK" in self.visibility: score -= 10
        return max(0, score)

    def is_in_ego_lane(self, box, w, h):
        x1, y1, x2, y2 = map(int, box)
        # Vehicle "footprint" is bottom center
        cx = (x1 + x2) // 2
        cy = y2

        # 10m line height
        y_10m = int(0.50 * h + self.focal_length * (self.cam_height / 10.0))
        
        if cy < y_10m: return False # Too far
        
        progress = (h - cy) / float(h - y_10m) # 0 at bottom, 1 at 10m line
        
        # Bottom width span was 0.05w to 0.95w (90% width)
        # Reduced by 40%: 90 * 0.6 = 54% width. Center is 0.5. 
        # So 0.5 +/- 0.27 -> 0.23w to 0.77w
        # Top width span at 10m marker: Keep it the same width as the previous narrowed parameter (0.4475w to 0.5525w) for a straighter path look.
        left_bound = (0.23 * w) + ( (0.4475 * w) - (0.23 * w) ) * progress
        right_bound = (0.77 * w) + ( (0.5525 * w) - (0.77 * w) ) * progress
        
        return left_bound < cx < right_bound

    def draw_augmented_grid(self, frame):
        h, w = frame.shape[:2]
        horizon = int(0.50 * h)
        
        y_10m = int(horizon + self.focal_length * (self.cam_height / 10.0))
        
        # --- Ego Lane Visual Corridor ---
        pt1_left = (int(0.23 * w), h)
        pt2_left = (int(0.4475 * w), y_10m)
        pt1_right = (int(0.77 * w), h)
        pt2_right = (int(0.5525 * w), y_10m)
        
        # 1. Subtle Polygon Fill
        overlay_poly = frame.copy()
        poly_pts = np.array([pt1_left, pt2_left, pt2_right, pt1_right], np.int32)
        cv2.fillPoly(overlay_poly, [poly_pts], (0, 0, 255))
        # 0.05 weighting makes it practically a whisper of red on the tarmac
        frame = cv2.addWeighted(overlay_poly, 0.05, frame, 0.95, 0)
        
        # 2. Crisp but Low-Opacity Boundary Lines
        overlay_lines = frame.copy()
        cv2.line(overlay_lines, pt1_left, pt2_left, (0, 0, 255), 2)
        cv2.line(overlay_lines, pt1_right, pt2_right, (0, 0, 255), 2)
        frame = cv2.addWeighted(overlay_lines, 0.4, frame, 0.6, 0) # 40% opacity for lines
        
        distances = [10, 20, 30, 50, 100]
        overlay = frame.copy()
        for dist in distances:
            pixel_offset = self.focal_length * (self.cam_height / dist)
            y_pos = int(horizon + pixel_offset)
            if y_pos < h and y_pos > horizon:
                cv2.line(overlay, (int(w*0.2), y_pos), (int(w*0.8), y_pos), (200, 200, 200), 1)
                # Minimal text
                cv2.putText(overlay, f"{dist}m", (int(w*0.8)+5, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        return frame

    def detect_lane_lines(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny Filter
        edges = cv2.Canny(blur, 50, 150)
        
        # ROI: Trapezoid for road
        h, w = frame.shape[:2]
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, h),
            (int(w * 0.40), int(h * 0.55)),
            (int(w * 0.60), int(h * 0.55)),
            (w, h)
        ]], np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Hough Transform
        lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=40, minLineLength=50, maxLineGap=150)
        
        valid_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = map(int, line.flatten())
                # Filter by slope (reject horizontalish lines)
                if x2 - x1 == 0: continue
                slope = (y2 - y1) / (x2 - x1)
                
                # Must be somewhat vertical
                if abs(slope) > 0.4:
                    valid_lines.append([x1, y1, x2, y2])
        
        return valid_lines

    def estimate_lane_count(self, boxes, w, detected_lines):
        # Combine Vehicle detections with Line detections
        
        # 1. Vehicle Clusters
        x_centers = sorted([(box[0] + box[2]) / 2 for box in boxes])
        veh_clusters = 0
        if x_centers:
            veh_clusters = 1
            last_x = x_centers[0]
            lane_threshold = w * 0.15
            for i in range(1, len(x_centers)):
                if x_centers[i] - last_x > lane_threshold:
                    veh_clusters += 1
                    last_x = x_centers[i]
        
        # 2. Line Detection
        # Count significant vertical lines
        line_count = len(detected_lines)
        # Crude estimation: 2 lines approx 1 lane, 3 lines 2 lanes, 4 lines 3 lanes etc.
        # But Hough lines are fragmented. Better to cluster the lines x-intercepts?
        # For simplicity, if we see many lines, it supports higher lane count.
        
        # Fusion Logic
        estimated_lanes = 1
        if veh_clusters > 1: estimated_lanes = veh_clusters
        
        # If we detect a lot of lines spread out, bump up the valid lane count confidence
        if line_count > 4 and estimated_lanes == 1:
            estimated_lanes = 2 # Suspect at least 2 lanes if we see lines
            
        self.lane_history.append(estimated_lanes)
        avg_lanes = sum(self.lane_history) / len(self.lane_history)
        
        if avg_lanes <= 1.5: return "1-LN"
        elif avg_lanes <= 2.5: return "2-LN"
        elif avg_lanes <= 3.5: return "3-LN"
        else: return "4+LN"

    def clean_memory(self, current_time):
        expired_ids = []
        for tid, timestamp in self.last_seen.items():
            if current_time - timestamp > 2.0: expired_ids.append(tid)
        for tid in expired_ids:
            if tid in self.kalman_filters: del self.kalman_filters[tid]
            if tid in self.track_lifetime: del self.track_lifetime[tid]
            if tid in self.lateral_history: del self.lateral_history[tid]
            if tid in self.advanced_physics_history: del self.advanced_physics_history[tid]
            if tid in self.smoother.boxes: del self.smoother.boxes[tid]
            if hasattr(self, 'intensity_history') and tid in self.intensity_history: del self.intensity_history[tid]
            if hasattr(self, 'signal_history') and tid in self.signal_history: del self.signal_history[tid]
            del self.last_seen[tid]

    def check_blind_spots(self, frame, boxes):
        h, w = frame.shape[:2]
        left_zone = int(w * 0.10)
        right_zone = int(w * 0.90)
        left_BS, right_BS = False, False
        for box in boxes:
            x1, y1, x2, y2 = box
            if y2 > h * 0.7:
                if x1 < left_zone: left_BS = True
                if x2 > right_zone: right_BS = True

        # FIX: Red Lines (Strips) instead of Dots
        if left_BS:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (20, h), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        if right_BS:
            overlay = frame.copy()
            cv2.rectangle(overlay, (w-20, 0), (w, h), (0, 0, 255), -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        return frame

    def analyze_traffic_light(self, frame, box):
        x1, y1, x2, y2 = box
        
        # Sanity check for minimum dimension
        if (x2 - x1) < 5 or (y2 - y1) < 10:
            return (0, 255, 255), "LIGHT"
            
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return (0, 255, 255), "LIGHT"

        # 1. Advanced Brightness Detection: Find the exact glowing bulb
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Find absolute peak brightness location
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(l_channel)
        
        # If the peak is too dim, the light is probably off
        if maxVal < 140:
            return (200, 200, 200), "OFF"
            
        # 2. Extract a tight Region of Interest (ROI) around the glowing core
        cx, cy = maxLoc
        radius = max(3, int(min((x2 - x1), (y2 - y1)) * 0.25)) # Adaptive radius centered strictly on glowing element
        
        # Safely bound the tight ROI coordinates
        roi_h, roi_w = roi.shape[:2]
        tx1 = max(0, cx - radius)
        ty1 = max(0, cy - radius)
        tx2 = min(roi_w, cx + radius)
        ty2 = min(roi_h, cy + radius)
        
        tight_roi = roi[ty1:ty2, tx1:tx2]
        if tight_roi.size == 0: return (200, 200, 200), "OFF"
        
        # 3. Strict HSV Thresholding solely on the peak glowing area
        hsv = cv2.cvtColor(tight_roi, cv2.COLOR_BGR2HSV)
        
        mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([12, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_green = cv2.inRange(hsv, np.array([45, 100, 100]), np.array([95, 255, 255]))
        mask_yellow = cv2.inRange(hsv, np.array([15, 130, 130]), np.array([35, 255, 255]))
        
        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        
        maximum = max(red_pixels, green_pixels, yellow_pixels)
        
        # 4. Final Verdict Mapping
        if maximum < 4: 
            return (200, 200, 200), "OFF" # Validates it's uncolored light or a reflection
        
        if maximum == red_pixels: 
            return (0, 0, 255), "RED"    # Red physical light (BGR: 0,0,255)
        if maximum == green_pixels: 
            return (0, 255, 0), "GREEN"  # Green physical light
        if maximum == yellow_pixels: 
            return (0, 255, 255), "YELLOW"
        
        return (0, 255, 255), "LIGHT" 

    def detect_vehicle_signals(self, frame, box, track_id):
        # Initialize the time-series intensity tracking dict
        if not hasattr(self, 'intensity_history'):
            self.intensity_history = {}
            
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        if w < 30 or h < 30: return "STRAIGHT"

        # ROI: Focus on upper/middle/lower rear (tail lights can be anywhere from 20% to 80% height)
        lights_y1 = y1 + int(h * 0.20)
        lights_y2 = y1 + int(h * 0.80)
        if lights_y1 >= lights_y2: return "STRAIGHT"

        roi = frame[lights_y1:lights_y2, x1:x2]
        if roi.size == 0: return "STRAIGHT"

        # Split into Left and Right regions for turn signals
        roi_h, roi_w = roi.shape[:2]
        w_split = int(roi_w * 0.40) # Broad 40% slice on each side to ensure we catch edge-mounted blinkers
        
        left_roi = roi[:, :w_split]
        right_roi = roi[:, roi_w - w_split:]

        # Analysis Function - Highly precise color and brightness detection
        def get_light_score(img_part):
            if img_part.size == 0: return 0
            hsv = cv2.cvtColor(img_part, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img_part, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Brightness mask (Adaptive minimal bound to ensure it's glowing)
            bright_mask = (l_channel > 140)
            
            # Red masks
            mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([12, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([165, 50, 50]), np.array([180, 255, 255]))
            
            # Amber/Yellow mask
            mask3 = cv2.inRange(hsv, np.array([12, 80, 80]), np.array([35, 255, 255]))
            
            red_pixels = cv2.countNonZero(((mask1 | mask2) & bright_mask).astype(np.uint8))
            amber_pixels = cv2.countNonZero((mask3 & bright_mask).astype(np.uint8))
            
            # Heavily weight amber/yellow because turn signals are mostly amber, but brake is red
            return red_pixels + (amber_pixels * 3.0)

        l_val = get_light_score(left_roi)
        r_val = get_light_score(right_roi)
        
        # Normalize score over the area to make tracking completely distance-invariant
        area = max(1, w_split * roi_h)
        l_score = l_val / area
        r_score = r_val / area
        
        if track_id not in self.intensity_history:
            self.intensity_history[track_id] = {'L': deque(maxlen=24), 'R': deque(maxlen=24)}
            
        self.intensity_history[track_id]['L'].append(l_score)
        self.intensity_history[track_id]['R'].append(r_score)
        
        hist_L = list(self.intensity_history[track_id]['L'])
        hist_R = list(self.intensity_history[track_id]['R'])
        
        if len(hist_L) < 8:
            return "STRAIGHT" # Need minimal history to analyze blinkers
            
        # --- Advanced Time-Series Mathematical Analysis ---
        mean_L = np.mean(hist_L)
        mean_R = np.mean(hist_R)
        
        max_L, min_L = np.max(hist_L), np.min(hist_L)
        max_R, min_R = np.max(hist_R), np.min(hist_R)
        
        ptp_L = max_L - min_L # Peak-to-Peak (Amplitude of the flash)
        ptp_R = max_R - min_R
        
        # Hyperparameters for signal classification
        active_thresh = 0.01 # At least 1% of the box must be lit on peak
        blink_amplitude = 0.015 # The difference between ON state and OFF state must be significant
        
        # 1. Brake Detection - Both sides are ON and steady (Low amplitude/variance relative to their mean)
        if mean_L > active_thresh and mean_R > active_thresh:
            if abs(mean_L - mean_R) < max(mean_L, mean_R) * 0.7:
                if ptp_L < mean_L * 1.5 and ptp_R < mean_R * 1.5:
                    return "BRAKE"
                    
        # 2. Advanced Blinker Detection - High Peak-to-Peak amplitude (pulsing pattern)
        left_blinking = ptp_L > blink_amplitude and max_L > active_thresh
        right_blinking = ptp_R > blink_amplitude and max_R > active_thresh
        
        if left_blinking and not right_blinking:
            if max_L > max_R * 1.2:
                return "SIGNAL LEFT"
        elif right_blinking and not left_blinking:
            if max_R > max_L * 1.2:
                return "SIGNAL RIGHT"
        elif left_blinking and right_blinking: 
            # Hazard lights flash both sides
            return "BRAKE"
            
        return "STRAIGHT" 

    def analyze_advanced_turn_intent(self, track_id, map_x, map_y, speed_kmh, dt):
        """
        Calculates a physical turn probability score (0-100) based on Minimap BEV 
        trajectory and kinematic deceleration.
        Returns: (left_prob, right_prob)
        """
        if track_id not in self.advanced_physics_history:
            self.advanced_physics_history[track_id] = {
                'x_hist': deque(maxlen=15),
                'y_hist': deque(maxlen=15),
                'v_hist': deque(maxlen=15)
            }
            
        hist = self.advanced_physics_history[track_id]
        hist['x_hist'].append(map_x)
        hist['y_hist'].append(map_y)
        hist['v_hist'].append(speed_kmh)
        
        if len(hist['x_hist']) < 8:
            return 0.0, 0.0
            
        x_list = list(hist['x_hist'])
        v_list = list(hist['v_hist'])
        
        # 1. Physics: Lateral BEV Velocity
        dx_total = x_list[-1] - x_list[0]
        bev_lat_vel = dx_total / (dt * len(x_list)) # Px per second
        
        # 2. Kinematics: Deceleration check (Slowing down to turn)
        dv_total = v_list[-1] - v_list[0]
        is_slowing_for_turn = dv_total < -3.0 # Dropped at least 3km/h over window
        
        left_prob = 0.0
        right_prob = 0.0
        
        # Base translation onto score
        if bev_lat_vel < -15:
            left_prob = min(100.0, abs(bev_lat_vel) * 1.5)
        elif bev_lat_vel > 15:
            right_prob = min(100.0, abs(bev_lat_vel) * 1.5)
            
        # Kinematic turn deceleration multiplier
        if is_slowing_for_turn and (left_prob > 10 or right_prob > 10):
            left_prob *= 1.3
            right_prob *= 1.3
            
        # Sharp trajectory curve spike bonus
        recent_dx = x_list[-1] - x_list[-4]
        if recent_dx < -10 and left_prob > 0: left_prob += 20
        if recent_dx > 10 and right_prob > 0: right_prob += 20
        
        return min(100.0, left_prob), min(100.0, right_prob)

    def draw_minimap(self, frame, objects):
        import numpy as np
        h, w = frame.shape[:2]
        map_w = 200
        map_h = 240 # Taller map for accurate long-distance scaling

        top_left = (20, h - map_h - 20)
        bottom_right = (20 + map_w, h - 20)

        # Create a dedicated black canvas to guarantee all drawings stay perfectly inside bounds
        map_canvas = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        
        center_x = map_w // 2
        bottom_y = map_h - 15  # Origin point of ego vehicle
        
        # 1. Background (Minimal Dark Slate)
        map_canvas[:] = (20, 25, 25)
        
        # --- TRUE BEV (Bird's Eye View) HOMOGRAPHY ---
        # Define 4 points on the camera image (Road Plane)
        horizon = int(0.50 * h)
        src_pts = np.float32([
            [int(w * 0.40), horizon + 20],    # Top Left (vanishing point vicinity)
            [int(w * 0.60), horizon + 20],    # Top Right
            [int(w * 0.10), h],               # Bottom Left (near hood)
            [int(w * 0.90), h]                # Bottom Right
        ])
        
        # Define 4 points on the target BEV map (Top-Down Plane)
        # We shrink the top a bit to represent standard parallel lane lines
        dst_pts = np.float32([
            [center_x - 30, 0],               # Top Left of Map
            [center_x + 30, 0],               # Top Right of Map
            [0, bottom_y],                    # Bottom Left of Map
            [map_w, bottom_y]                 # Bottom Right of Map
        ])
        
        # Calculate Homography Matrix
        H_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # 2. Minimalist FOV Corridor / Lane Lines on BEV
        cv2.line(map_canvas, (center_x - 30, 0), (0, bottom_y), (70, 75, 75), 1)
        cv2.line(map_canvas, (center_x + 30, 0), (map_w, bottom_y), (70, 75, 75), 1)
        cv2.line(map_canvas, (center_x, 0), (center_x, bottom_y), (50, 55, 55), 1) # Center Line

        max_dist = 100.0
        scale = bottom_y / max_dist

        # 3. Horizontal Grid Lines
        for d in [20, 40, 60, 80]:
            y_d = bottom_y - int(d * scale)
            if y_d > 0:
                cv2.line(map_canvas, (0, y_d), (map_w, y_d), (60, 65, 65), 1)
                cv2.putText(map_canvas, f"{d}m", (map_w - 30, y_d - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 130, 130), 1)

        # 4. Clean Vector Ego Vehicle Symbol
        ego_pts = np.array([
            [center_x, bottom_y - 12],
            [center_x - 6, bottom_y + 4],
            [center_x, bottom_y - 2],
            [center_x + 6, bottom_y + 4]
        ], np.int32)
        cv2.fillPoly(map_canvas, [ego_pts], (0, 255, 0))

        # 5. Minimal Informative Blips (Inside Canvas)
        for obj in objects:
            if len(obj) >= 8:
                x1, y1, x2, y2, color, dist_m, is_relevant, track_id = obj[:8]
            elif len(obj) == 7:
                x1, y1, x2, y2, color, dist_m, is_relevant = obj
                track_id = -1
            else:
                x1, y1, x2, y2, color, dist_m = obj[:6] 
                is_relevant = False
                track_id = -1
            
            # Map object using Perspective Transform
            cx, cy = (x1 + x2) // 2, y2
            
            # Transform the (cx, cy) camera coordinate to Map coordinate using H_matrix
            pt = np.array([[[cx, cy]]], dtype=np.float32)
            transformed_pt = cv2.perspectiveTransform(pt, H_matrix)
            
            map_obj_x = int(transformed_pt[0][0][0])
            
            # Use combined Y logic: Transform Y provides lateral spread, but we keep our reliable depth estimation for Y placement
            map_obj_y = bottom_y - int(dist_m * scale)
            
            # Draw strictly within canvas coordinates
            if 0 <= map_obj_x < map_w and 0 <= map_obj_y < map_h:
                if is_relevant:
                    cv2.circle(map_canvas, (map_obj_x, map_obj_y), 4, color, -1)
                    cv2.circle(map_canvas, (map_obj_x, map_obj_y), 6, (0, 0, 255), 1)
                else:
                    cv2.circle(map_canvas, (map_obj_x, map_obj_y), 2, color, -1)

        # 6. Perfect Clipping Application
        # Extract the region of interest from the frame
        roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        # Blend the isolated map canvas on top natively
        blended = cv2.addWeighted(roi, 0.15, map_canvas, 0.85, 0)
        frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = blended

        # 7. Minimalist 1px Frame Border (No glow, perfectly constrained)
        cv2.rectangle(frame, top_left, bottom_right, (100, 110, 110), 1)

        return frame

    def draw_hud(self, frame, mode_text, lane_type, safety_score, live_stats, active_signs):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        self.draw_text_with_outline(frame, f"ADAS V34.2 | {lane_type} | {self.weather_status}", (10, 20), 0.4, (255, 255, 255))
        self.draw_text_with_outline(frame, f"{int(self.fps)} FPS | {int(self.inference_time)}ms", (w-120, 20), 0.4, (200, 200, 200))

        score_color = (0, 255, 0)
        if safety_score < 60: score_color = (0, 0, 255)
        elif safety_score < 85: score_color = (0, 165, 255)

        cv2.line(frame, (w//2 - 50, 28), (w//2 + 50, 28), (100, 100, 100), 4)
        bar_len = int((safety_score/100) * 100)
        cv2.line(frame, (w//2 - 50, 28), (w//2 - 50 + bar_len, 28), score_color, 4)

        # --- FIX: LIVE TRAFFIC STATS PANEL ---
        panel_w = 140
        panel_h = 200 # Height increased for extra icons
        panel_x = w - panel_w - 10
        panel_y = 60

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 1)

        self.draw_text_with_outline(frame, "TRAFFIC", (panel_x + 35, panel_y + 20), 0.5, (200, 200, 200))
        cv2.line(frame, (panel_x + 10, panel_y + 25), (panel_x + panel_w - 10, panel_y + 25), (100, 100, 100), 1)

        y_off = panel_y + 45
        icons = {'car': 'CAR', 'person': 'PED', 'truck': 'TRK', 'bus': 'BUS', 'motorcycle': 'MOTO', 'bicycle': 'BIKE', 'train': 'TRN'}
        for key, label in icons.items():
            # Show live counts, default to 0 if not present
            count = live_stats.get(key, 0)
            color = self.colors.get(key, (255, 255, 255))
            cv2.putText(frame, f"{label}:", (panel_x + 15, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"{count}", (panel_x + 100, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_off += 20

        state_y = 60
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, state_y), (150, state_y + 100), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.rectangle(frame, (10, state_y), (150, state_y + 100), (100, 100, 100), 1)

        self.draw_text_with_outline(frame, "STATUS", (45, state_y + 20), 0.5, (200, 200, 200))
        cv2.line(frame, (20, state_y + 25), (140, state_y + 25), (100, 100, 100), 1)
        self.draw_text_with_outline(frame, f"WTHR: {self.weather_status}", (20, state_y + 45), 0.4, (255, 255, 255))
        vis_color = (0, 255, 0) if "GOOD" in self.visibility else (0, 165, 255)
        self.draw_text_with_outline(frame, f"VIS: {self.visibility}", (20, state_y + 65), 0.4, vis_color)
        grip_color = (0, 255, 0) if self.friction_coef > 0.6 else (0, 0, 255)
        self.draw_text_with_outline(frame, f"GRIP: {int(self.friction_coef*100)}%", (20, state_y + 85), 0.4, grip_color)

        if mode_text:
            self.draw_text_with_outline(frame, mode_text, (200, h-15), 0.5, (100, 100, 255))
            self.draw_text_with_outline(frame, self.beam_suggestion, (350, h-15), 0.5, (255, 255, 0))

        # --- DRAW ACTIVE ROAD SIGNS & ALERTS ON HUD ---
        if getattr(self, 'pothole_alert', False):
            active_signs.append(("POTHOLE DETECTED!", (0, 0, 255)))
        if getattr(self, 'ldw_warning', None):
            active_signs.append((self.ldw_warning, (0, 0, 255)))
            
        if active_signs:
            sign_y = panel_y + panel_h + 20
            # Draw semi-transparent panel
            sign_overlay = frame.copy()
            cv2.rectangle(sign_overlay, (panel_x, sign_y), (panel_x + panel_w, sign_y + max(50, len(active_signs)*30 + 30)), (20, 20, 20), -1)
            frame = cv2.addWeighted(sign_overlay, 0.7, frame, 0.3, 0)
            cv2.rectangle(frame, (panel_x, sign_y), (panel_x + panel_w, sign_y + max(50, len(active_signs)*30 + 30)), (0, 165, 255), 1)
            
            self.draw_text_with_outline(frame, "ROAD SIGNS", (panel_x + 25, sign_y + 20), 0.5, (0, 165, 255))
            cv2.line(frame, (panel_x + 10, sign_y + 25), (panel_x + panel_w - 10, sign_y + 25), (0, 165, 255), 1)
            
            y_offset = sign_y + 45
            for sign_label, sign_color in active_signs:
                cv2.circle(frame, (panel_x + 20, y_offset - 4), 6, sign_color, -1)
                cv2.putText(frame, sign_label, (panel_x + 35, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                y_offset += 25

        return frame

    def process_frame(self, frame):
        current_time = time.time()
        dt = current_time - self.prev_frame_time
        self.prev_frame_time = current_time
        if dt == 0: dt = 0.033

        current_vehicle_count = 0
        h, w = frame.shape[:2]

        horizon_level = int(h * 0.45)
        road_contour = np.array([[0, h], [w, h], [int(w * 0.70), horizon_level], [int(w * 0.30), horizon_level]], dtype=np.int32)

        t0 = time.time()
        # FIX: half=False required for some local GPUs (GTX 16xx series) to avoid 0 detections
        # FIX: Use ByteTrack to avoid BoT-SORT 'not positive definite' / 'matching points' errors
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.25, iou=0.5, agnostic_nms=True, verbose=False, device=0, half=False)
        t1 = time.time()
        self.inference_time = (t1 - t0) * 1000

        detected_boxes = []
        minimap_objects = []
        vehicle_boxes_only = []
        active_road_signs = [] # Track detected signs in this frame
        max_risk_prob = 0

        # New: Live Frame Stats Dictionary
        live_stats = {k: 0 for k in ['car', 'person', 'truck', 'bus', 'motorcycle', 'bicycle', 'train']}

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train']:
                    vehicle_boxes_only.append([x1, y1, x2, y2])
                    current_vehicle_count += 1

                # Update live stats
                if label in live_stats:
                    live_stats[label] += 1

        if self.frame_count % 30 == 0:
            self.clean_memory(current_time)
            
        if self.frame_count % 5 == 0:
            self.analyze_environment(frame, current_vehicle_count)

        # Lane Detection
        detected_lines = self.detect_lane_lines(frame)
        lane_type = self.estimate_lane_count(vehicle_boxes_only, w, detected_lines)
        
        # --- LDW (Lane Departure Warning) Logic ---
        self.ldw_warning = None
        if len(detected_lines) >= 2:
            intercepts = []
            for line in detected_lines:
                lx1, ly1, lx2, ly2 = line
                if ly2 != ly1:
                    slope = (lx2 - lx1) / (ly2 - ly1)
                    x_int = lx1 + slope * (h - ly1)
                    intercepts.append(x_int)
            if intercepts:
                intercepts.sort()
                center_x = w / 2
                left_lines = [x for x in intercepts if x < center_x]
                right_lines = [x for x in intercepts if x > center_x]
                
                if left_lines and right_lines:
                    left_int = left_lines[-1]
                    right_int = right_lines[0]
                    lane_width = right_int - left_int
                    if lane_width > w * 0.15: # Valid lane width
                        if center_x < left_int + lane_width * 0.15:
                            self.ldw_warning = "<- DEPARTURE WARNING"
                        elif center_x > right_int - lane_width * 0.15:
                            self.ldw_warning = "DEPARTURE WARNING ->"
        
        # Shape-based Traffic Sign Detection (OpenCV)
        hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_r = cv2.inRange(hsv_full, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask_r2 = cv2.inRange(hsv_full, np.array([160, 100, 100]), np.array([180, 255, 255]))
        mask_y = cv2.inRange(hsv_full, np.array([15, 100, 100]), np.array([35, 255, 255]))
        cnts, _ = cv2.findContours(cv2.GaussianBlur(mask_r | mask_r2 | mask_y, (5, 5), 0), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if cv2.contourArea(cnt) > 600:
                approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
                if len(approx) == 8:
                    x, y, cw, ch = cv2.boundingRect(approx)
                    active_road_signs.append(("STOP SIGN", (0, 0, 255)))
                    cv2.putText(frame, "STOP", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif len(approx) == 3:
                    x, y, cw, ch = cv2.boundingRect(approx)
                    active_road_signs.append(("YIELD / WARNING", (0, 255, 255)))

        frame, mode_text = self.apply_night_vision(frame, vehicle_boxes_only)
        frame = self.draw_augmented_grid(frame)

        critical_obj_center = None
        min_ttc_val = 100.0

        # Create localized Minimap Transform inside process_frame for Advanced Physics engine mapping
        map_w, map_h = 200, 240
        map_center_x, map_bottom_y = map_w // 2, map_h - 15
        
        src_pts = np.float32([
            [int(w * 0.40), int(h * 0.50) + 20],
            [int(w * 0.60), int(h * 0.50) + 20],
            [int(w * 0.10), h],
            [int(w * 0.90), h]
        ])
        dst_pts = np.float32([
            [map_center_x - 30, 0],
            [map_center_x + 30, 0],
            [0, map_bottom_y],
            [map_w, map_bottom_y]
        ])
        H_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        map_scale = map_bottom_y / 100.0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                raw_x1, raw_y1, raw_x2, raw_y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                track_id = int(box.id[0]) if box.id is not None else -1

                x1, y1, x2, y2 = self.smoother.update(track_id, [raw_x1, raw_y1, raw_x2, raw_y2]) if track_id != -1 else [raw_x1, raw_y1, raw_x2, raw_y2]

                contact_point = (int((x1 + x2) / 2), int(y2))
                is_on_road = cv2.pointPolygonTest(road_contour, contact_point, False) >= 0
                if label in ['traffic light', 'stop sign']: is_on_road = True

                if is_on_road and label in self.colors:
                    detected_boxes.append([x1, y1, x2, y2])
                    color = self.colors.get(label, (0, 255, 255))

                    if label == 'traffic light':
                        tl_color, tl_text = self.analyze_traffic_light(frame, [x1, y1, x2, y2])
                        color = tl_color
                        if tl_text:
                            cv2.putText(frame, tl_text, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            active_road_signs.append((f"TL: {tl_text}", color))
                            
                    elif label == 'stop sign':
                        cv2.putText(frame, "STOP", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        active_road_signs.append(("STOP SIGN", (0, 0, 255)))

                    w_box = x2 - x1
                    raw_dist = self.estimate_distance_advanced(y2, h, w_box, label)
                    stable_dist, velocity, accel = raw_dist, 0, 0
                    is_cut_in = False
                    intent_text = "STRAIGHT"

                    if track_id != -1:
                        self.last_seen[track_id] = current_time
                        if track_id not in self.tracked_ids: self.tracked_ids.add(track_id)

                        stable_dist, velocity, accel = self.update_kalman(track_id, raw_dist, dt)
                        is_cut_in, lat_vel, intent_text = self.detect_lateral_intent(track_id, (x1+x2)/2, dt, w)
                        self.update_trajectory(track_id, ((x1+x2)//2, (y1+y2)//2))
                        
                        # --- PREDICTIVE TRAJECTORY TAIL (Backend Only) ---
                        # self.object_paths populated via self.update_trajectory
                        # No UI drawing to keep HUD minimal.

                    # Determine relevance early for minimap
                    in_ego_lane = self.is_in_ego_lane([x1, y1, x2, y2], w, h)
                    is_relevant = False
                    if self.track_lifetime.get(track_id, 0) > 5 or is_cut_in:
                        is_relevant = in_ego_lane or is_cut_in

                    minimap_objects.append([x1, y1, x2, y2, color, stable_dist, is_relevant, track_id])

                    if self.track_lifetime.get(track_id, 0) > 5 or is_cut_in:
                        
                        # Color Coding
                        if not is_relevant: color = (0, 255, 255) # Yellow/Reference
                        if is_cut_in: color = (0, 165, 255) # Orange
                        if is_relevant: color = (0, 255, 0) # Green (default for monitored)

                        l_len = int((x2-x1) * 0.2)
                        cv2.line(frame, (x1, y1), (x1+l_len, y1), color, 2)
                        cv2.line(frame, (x1, y1), (x1, y1+l_len), color, 2)
                        cv2.line(frame, (x2, y2), (x2-l_len, y2), color, 2)
                        cv2.line(frame, (x2, y2), (x2, y2-l_len), color, 2)

                        # --- FIX: Skip Physics/Info for Traffic Light ---
                        if label != 'traffic light':
                            speed_kmh = velocity * 3.6
                            ttc, headway, is_braking, impact_severity = self.calculate_physics_metrics(velocity, stable_dist, accel)
                            stop_dist_req = self.calculate_stopping_distance(abs(speed_kmh))
                            risk_prob = self.calculate_collision_risk(ttc)
                            # Advanced adjustment to risk based on impact severity
                            risk_prob = min(100, int(risk_prob + impact_severity * 10))
                            max_risk_prob = max(max_risk_prob, risk_prob)

                            # Initialize label_text here to fix NameError
                            label_text = f"{label[:3].upper()}"

                            # --- ACCURATE CROSS TRAFFIC ALERT ---
                            is_cross_traffic = False
                            if label in ['car', 'truck', 'bus'] and not in_ego_lane:
                                obj_aspect = (x2 - x1) / max(1, y2 - y1)
                                cx = (x1 + x2) // 2
                                moving_to_center = (cx < w//2 and lat_vel > 15) or (cx > w//2 and lat_vel < -15)
                                if moving_to_center and obj_aspect > 1.2 and ttc < 5.0:
                                    is_cross_traffic = True

                            # --- MULTI-TIER FCW OVERLAY ---
                            if is_relevant and ttc < 5.0 and not is_cut_in:
                                # Watch -> Warning -> Critical
                                box_color = (0, 255, 255) # Yellow
                                if ttc < 3.0: box_color = (0, 165, 255) # Orange
                                if ttc < 1.5: box_color = (0, 0, 255) # Red Critical
                                # Pulsing effect
                                if ttc < 3.0 and int(time.time() * 10) % 2 == 0:
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)

                            if is_relevant:
                                info_text = f"{stable_dist:.0f}m"
                                if abs(speed_kmh) > 5: info_text += f" {int(abs(speed_kmh))}kmh"

                                text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                                cv2.rectangle(frame, (x1, y1-20), (x1 + text_size[0] + 10, y1), color, -1)
                                cv2.putText(frame, info_text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
                            else:
                                cv2.putText(frame, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                            # --- FIX: ONCOMING VS EGO-DIRECTION TRACKING ---
                            # Track distance changes over time to see if it's getting closer (ONCOMING) or staying/moving away
                            vehicle_direction = "EGO-DIR" # Default assume same direction
                            if track_id != -1:
                                if track_id not in self.direction_history:
                                    self.direction_history[track_id] = deque(maxlen=10) # Track last 10 frames of distance
                                self.direction_history[track_id].append(stable_dist)
                                
                                dist_hist = list(self.direction_history[track_id])
                                if len(dist_hist) > 5:
                                    # If the distance is rapidly decreasing and it's not in our exact lane, it's likely oncoming
                                    dist_change = dist_hist[-1] - dist_hist[0]
                                    if dist_change < -2.0 and not in_ego_lane: 
                                        vehicle_direction = "ONCOMING"
                                        
                            # --- ULTIMATE INTENT COMBINER ---
                            # Combine physical lateral shifting with optical turn signals
                            if is_relevant and label in ['car', 'truck', 'bus'] and track_id != -1:
                                
                                # Superior Intent & Light State Interpreter
                                sig = "STRAIGHT"
                                if vehicle_direction != "ONCOMING":
                                    sig_result = self.detect_vehicle_signals(frame, [x1, y1, x2, y2], track_id)
                                    if sig_result: sig = sig_result
                                
                                # Process minimap trajectory coordinates for Advanced Physics
                                cx, cy = (x1 + x2) // 2, y2
                                pt = np.array([[[cx, cy]]], dtype=np.float32)
                                transformed_pt = cv2.perspectiveTransform(pt, H_matrix)
                                map_obj_x = int(transformed_pt[0][0][0])
                                map_obj_y = map_bottom_y - int(stable_dist * map_scale)
                                
                                phys_l_prob, phys_r_prob = self.analyze_advanced_turn_intent(track_id, map_obj_x, map_obj_y, abs(speed_kmh), dt)

                                # Calculate Ultimate Confidences
                                final_score_l, final_score_r = 0.0, 0.0
                                
                                # 1. Signal Multipliers (Optical Intent is very strong)
                                if sig == "SIGNAL LEFT": final_score_l += 60.0
                                if sig == "SIGNAL RIGHT": final_score_r += 60.0
                                
                                # 2. Camera Shifting Intent (Immediate frame-to-frame shift)
                                if intent_text == "SHIFT LEFT": final_score_l += 30.0
                                if intent_text == "SHIFT RIGHT": final_score_r += 30.0
                                
                                # 3. Advanced BEV Minimap Physics (Trajectory & Kinematics)
                                final_score_l += phys_l_prob
                                final_score_r += phys_r_prob

                                final_intent = None
                                sig_color = (0, 255, 0) # Base Safe Green
                                
                                # Absolute Decision Matrix
                                if sig == "BRAKE" or is_braking:
                                    final_intent = "BRAKING"
                                    sig_color = (0, 0, 255) # Red danger
                                    is_braking = True
                                elif final_score_l > 85.0:
                                    final_intent = "FINAL: TURN LEFT"
                                    sig_color = (0, 165, 255)
                                    cv2.putText(frame, "TRACKING L-TURN", (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
                                elif final_score_r > 85.0:
                                    final_intent = "FINAL: TURN RIGHT"
                                    sig_color = (0, 165, 255)
                                    cv2.putText(frame, "TRACKING R-TURN", (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
                                else:
                                    # Fallbacks
                                    if sig == "SIGNAL LEFT" and intent_text == "SHIFT LEFT":
                                        final_intent = "<- CUTTING"
                                        sig_color = (0, 165, 255) # Orange warning
                                    elif sig == "SIGNAL RIGHT" and intent_text == "SHIFT RIGHT":
                                        final_intent = "CUTTING ->"
                                        sig_color = (0, 165, 255)
                                    elif sig == "SIGNAL LEFT":
                                        final_intent = "<- LEFT"
                                        sig_color = (0, 165, 255)
                                    elif sig == "SIGNAL RIGHT":
                                        final_intent = "RIGHT ->"
                                        sig_color = (0, 165, 255)
                                    elif intent_text != "STRAIGHT":
                                        final_intent = f"{intent_text}"
                                        sig_color = (150, 150, 150)
                                    
                                if final_intent:
                                    cv2.putText(frame, final_intent, (x1, y2+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, sig_color, 2)
                                    
                                # Optional: Visual tag for oncoming traffic filtering
                                if vehicle_direction == "ONCOMING":
                                    cv2.putText(frame, "ONC", (x1+30, y2+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                            y_offset = y1 - 25
                            if is_cross_traffic:
                                cv2.putText(frame, "CROSSING", (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                y_offset -= 15
                            elif is_cut_in: 
                                cv2.putText(frame, "CUT-IN", (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                y_offset -= 15
                                
                            if is_relevant and is_braking:
                                cv2.putText(frame, "BRK", (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                y_offset -= 15
                                color = (0, 0, 255)

                            if is_relevant and ttc < 3.0:
                                if risk_prob > 50:
                                    cv2.putText(frame, f"RISK {risk_prob}%", (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    color = (0, 0, 255)
                                if ttc < min_ttc_val:
                                    min_ttc_val = ttc
                                    critical_obj_center = ((x1+x2)//2, y2)

                            if is_relevant and stable_dist < stop_dist_req and stable_dist < 50:
                                 cv2.putText(frame, "CLOSE", (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            width = x2 - x1
                            height = y2 - y1
                            
                            # --- PEDESTRIAN INTENT LOGIC ---
                            if label in ['person', 'bicycle', 'motorcycle', 'dog', 'cow'] and stable_dist < 25:
                                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                                radius = max(width, y2-y1) // 2 + 10
                                
                                # Analyze aspect ratio to determine intent
                                aspect_ratio = width / max(1, height)
                                
                                # An aspect ratio > 0.45 often means walking/facing sideways (across the road)
                                # A High lateral velocity means they are actively moving across the view
                                crossing_intent = False
                                if label == 'person':
                                    if aspect_ratio > 0.45 or abs(lat_vel) > 10:
                                        crossing_intent = True
                                
                                if crossing_intent and not in_ego_lane:
                                    # Yellow pre-warning instead of red panic
                                    if int(time.time() * 5) % 2 == 0:
                                        cv2.circle(frame, (center_x, center_y), radius, (0, 165, 255), 2)
                                        cv2.putText(frame, "CROSSING INTENT", (center_x-40, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                                elif in_ego_lane or stable_dist < 10:
                                    # Red Critical Warning
                                    if int(time.time() * 8) % 2 == 0:
                                        cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), 3)
                                        cv2.putText(frame, "WATCH OUT!", (center_x-40, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                else:
                                    # Safe state monitoring
                                    cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 1)

                            if label in ['dog', 'cow', 'horse', 'sheep']:
                                cv2.putText(frame, "ANIMAL!", (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 105, 255), 2)

                            if label == 'stop sign' and stable_dist < 30:
                                cv2.putText(frame, "STOP", (w//2 - 50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                            # Move draw_text inside the if block or ensure label_text is defined
                            self.draw_text_with_outline(frame, label_text, (x1, y1-5), 0.5, color, 1)

        if critical_obj_center:
            cv2.circle(frame, critical_obj_center, 15, (0, 0, 255), 2)

        # Blind Spot: Red Strips
        frame = self.check_blind_spots(frame, detected_boxes)
        # lane_type already calculated above
        safety_score = self.calculate_safety_score(min_ttc_val, self.weather_status, self.road_status)

        # Original 2D Minimap
        frame = self.draw_minimap(frame, minimap_objects)
        # Pass Live Stats & Road Signs to HUD
        frame = self.draw_hud(frame, mode_text, lane_type, safety_score, live_stats, active_road_signs)

        return frame

# 5. EXECUTION
# -----------------------------------
print("Initializing ADAS V35.0 (LOCAL VERSION) WITH NEXT-GEN FEATURES...")
processor = ADASProcessor()

# Initialize Webcam
print(f"Opening camera input: {VIDEO_SOURCE} with Threaded Video Stream...")
cap = ThreadedVideoStream(VIDEO_SOURCE).start()

# Allow camera buffer to warm up
time.sleep(1.0)

print("Camera opened and buffered successfully. Starting Dashboard...")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.005) # Prevent 100% CPU lock when queue is empty!
        continue # Wait for next threaded frame

    try:
        # Process frame
        final_frame = processor.process_frame(frame)

        # Display result
        cv2.imshow('ADAS V35.0 PRO', final_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

cap.release()
cv2.destroyAllWindows()