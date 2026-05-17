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

print("Loading ULTRA POWER Model (YOLO11x) for Maximum Accuracy & Efficiency...")
# Using the most powerful YOLO11 model as requested
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
        self.smoother = BoxSmoother(alpha=0.5)

        # Lane History
        self.lane_history = deque(maxlen=20)

        # Statistics
        self.stats = { 'car': 0, 'person': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0 }
        self.tracked_ids = set()

        # Signal Tracking History
        # Format: { track_id: [ (timestamp, 'LEFT'/'RIGHT'/'BRAKE'/'OFF'), ... ] }
        self.signal_history = {}

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
        if road_roi.size > 0:
            mean, std_dev = cv2.meanStdDev(road_roi)
            if std_dev[0][0] > 50: self.road_status = "BUMPY"
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

    def estimate_distance_geometric(self, y_bottom, h):
        if y_bottom >= h: return 0.0
        horizon = 0.5 * h
        if y_bottom <= horizon: return 150.0
        dy = y_bottom - horizon
        alpha = np.arctan(dy / self.focal_length)
        dist = self.cam_height / np.tan(alpha)
        return dist

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

    def detect_cut_in(self, track_id, current_x, dt, width):
        is_cut_in = False
        center_screen = width / 2
        if track_id not in self.lateral_history:
            self.lateral_history[track_id] = {'x': current_x, 'v': 0}
            return False, 0
        prev = self.lateral_history[track_id]
        dx = current_x - prev['x']
        lat_velocity = dx / dt
        self.lateral_history[track_id] = {'x': current_x, 'v': lat_velocity}
        moving_to_center = False
        if current_x < center_screen and lat_velocity > 10: moving_to_center = True
        if current_x > center_screen and lat_velocity < -10: moving_to_center = True
        if moving_to_center and abs(lat_velocity) > 50: is_cut_in = True
        return is_cut_in, lat_velocity

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

        # Define Ego Lane trapezoid (approximate/heuristic if lines not perfect)
        # Bottom: 20% to 80% width
        # Top (Horizon): 45% to 55% width
        
        # Simple point-in-triangle/trapezoid check
        # Left boundary line: (0.2*w, h) -> (0.45*w, 0.45*h)
        # Right boundary line: (0.8*w, h) -> (0.55*w, 0.45*h)
        
        # Calculate expected x-range at this y-level
        # Normalized y from bottom (0.0) to horizon (1.0)
        horizon_y = 0.45 * h
        if cy < horizon_y: return False # Too far/above horizon
        
        relative_y = (h - cy) / (h - horizon_y) # 0 at bottom, 1 at horizon (Wait, inverse?)
        # Let's do interpolation
        progress = (h - cy) / (h - horizon_y) # 0 at bottom, 1 at horizon
        
        # Width of lane tapers from W_bottom to W_top
        # Bottom width span: 0.15w to 0.85w (generous)
        # Top width span: 0.45w to 0.55w
        
        left_bound = (0.2 * w) + ( (0.45 * w) - (0.2 * w) ) * progress
        right_bound = (0.8 * w) + ( (0.55 * w) - (0.8 * w) ) * progress
        
        return left_bound < cx < right_bound

    def draw_augmented_grid(self, frame):
        h, w = frame.shape[:2]
        horizon = int(0.50 * h)
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
            (int(w * 0.4), int(h * 0.55)),
            (int(w * 0.6), int(h * 0.55)),
            (w, h)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
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
                if abs(slope) > 0.4: # Must be somewhat vertical
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
            if tid in self.smoother.boxes: del self.smoother.boxes[tid]
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
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return (0, 0, 0), ""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])) + \
                   cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        mask_green = cv2.inRange(hsv, np.array([40, 70, 50]), np.array([90, 255, 255]))
        red_px = cv2.countNonZero(mask_red)
        green_px = cv2.countNonZero(mask_green)
        if red_px > green_px and red_px > 20: return (0, 0, 255), "STOP"
        elif green_px > red_px and green_px > 20: return (0, 255, 0), "GO"
        else: return (0, 255, 255), "LIGHT"

    def detect_vehicle_signals(self, frame, box, track_id):
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        if w < 30 or h < 30: return None

        # ROI: Focus on upper-middle rear (tail lights)
        # Avoid the very bottom (exhaust/bumper reflectance)
        # Typical tail lights are in the 30%-60% height range of the car box
        lights_y1 = y1 + int(h * 0.35)
        lights_y2 = y1 + int(h * 0.65)
        if lights_y1 >= lights_y2: return None

        roi = frame[lights_y1:lights_y2, x1:x2]
        if roi.size == 0: return None

        # Split into Left and Right regions (outer 25%)
        roi_h, roi_w = roi.shape[:2]
        w_split = int(roi_w * 0.30)
        
        left_roi = roi[:, :w_split]
        right_roi = roi[:, roi_w - w_split:]

        # Analysis Function
        def get_light_intensity(img_part):
            # HSV for color (Red/Orange) + LAB for Brightness
            hsv = cv2.cvtColor(img_part, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img_part, cv2.COLOR_BGR2LAB)
            
            l_channel = lab[:, :, 0]
            
            # Red/Orange Masks
            # Red wraps around 180->0
            mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
            # Amber/Orange
            mask3 = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([30, 255, 255]))
            
            color_mask = mask1 | mask2 | mask3
            
            # Brightness Mask (Tail lights are bright)
            bright_mask = (l_channel > 140)
            
            # Combined
            final_mask = color_mask & bright_mask
            return cv2.countNonZero(final_mask.astype(np.uint8))

        l_val = get_light_intensity(left_roi)
        r_val = get_light_intensity(right_roi)
        
        # Check Saturation for Headlight Rejection (White light prevention)
        # If highly bright but low saturation -> Headlight -> Ignore
        # Get average saturation of the "bright" areas
        def check_saturation(img_part):
            hsv = cv2.cvtColor(img_part, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            # Consider only bright pixels
            bright_mask = (v > 200).astype(np.uint8)
            if cv2.countNonZero(bright_mask) > 10:
                mean_s = np.mean(s[bright_mask])
                return mean_s
            return 255 # Assume ok if no bright pixels
            
        l_sat = check_saturation(left_roi)
        r_sat = check_saturation(right_roi)
        
        # If saturation is very low (< 30), it's likely white light (headlight/street light)
        if l_sat < 40 and r_sat < 40:
             return None # Reject as headlight/reflection

        # Threshold relative to size
        threshold = (w_split * roi_h) * 0.02 # 2% of pixels must be "light"

        # Instantaneous State
        current_state = 'OFF'
        if l_val > threshold and r_val > threshold:
            current_state = 'BRAKE'
        elif l_val > threshold:
            current_state = 'LEFT'
        elif r_val > threshold:
            current_state = 'RIGHT'
            
        # --- Temporal Smoothing (Debounce & Blink Detection) ---
        if track_id not in self.signal_history:
            self.signal_history[track_id] = deque(maxlen=15) # ~0.5s history at 30fps
            
        self.signal_history[track_id].append(current_state)
        
        history = list(self.signal_history[track_id])
        
        # 1. BRAKE: Needs consistent ON state
        brake_count = history.count('BRAKE')
        if brake_count > len(history) * 0.5: # 50% frames are brake
            return "BRAKE"
            
        # 2. TURN SIGNALS: Check for blinking (OFF -> LEFT -> OFF ...) or consistent LEFT
        left_count = history.count('LEFT')
        right_count = history.count('RIGHT')
        
        # If we see mostly LEFT (or mixed LEFT/OFF but rhythmic), call it. 
        # Simple logic: if > 30% frames are LEFT and almost no RIGHT
        if left_count > len(history) * 0.3 and right_count < 2:
            return "LEFT TURN"
            
        if right_count > len(history) * 0.3 and left_count < 2:
            return "RIGHT TURN"

        return None

    def draw_minimap(self, frame, objects):
        h, w = frame.shape[:2]
        map_w = 200
        map_h = 200

        overlay = frame.copy()
        top_left = (20, h - map_h - 20)
        bottom_right = (20 + map_w, h - 20)
        cv2.rectangle(overlay, top_left, bottom_right, (15, 25, 30), -1) # Dark radar background
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2) # Glowing border

        center_x = top_left[0] + map_w // 2
        bottom_y = bottom_right[1] - 15
        
        # Real Ego Vehicle Icon
        ego_pts = np.array([
            [center_x, bottom_y - 15],
            [center_x - 8, bottom_y + 5],
            [center_x + 8, bottom_y + 5]
        ], np.int32)
        cv2.fillPoly(frame, [ego_pts], (0, 255, 0))

        # Field of View lines
        cv2.line(frame, (center_x, bottom_y), (top_left[0], top_left[1]), (50, 80, 90), 1)
        cv2.line(frame, (center_x, bottom_y), (bottom_right[0], top_left[1]), (50, 80, 90), 1)

        max_dist = 100.0 # Extended range
        scale = (map_h - 20) / max_dist

        for d in [25, 50, 75, 100]:
            y_ring = bottom_y - int(d * scale)
            if y_ring > top_left[1]:
                cv2.ellipse(frame, (center_x, bottom_y), (int(d * scale * 1.5), int(d*scale)), 0, 180, 360, (50, 80, 90), 1)
                cv2.putText(frame, f"{d}m", (center_x+2, y_ring-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 200, 200), 1)

        for obj in objects:
            if len(obj) == 7:
                x1, y1, x2, y2, color, dist_m, is_relevant = obj
            else:
                x1, y1, x2, y2, color, dist_m = obj
                is_relevant = False
            
            cx, cy = (x1 + x2) // 2, y2
            rel_x = (cx - (w/2)) / (w/2)
            map_x_offset = int(rel_x * (map_w / 1.5)) # Wider spread for realism
            map_obj_x = center_x + map_x_offset
            map_obj_y = bottom_y - int(dist_m * scale)

            map_obj_x = np.clip(map_obj_x, top_left[0]+5, bottom_right[0]-5)
            map_obj_y = np.clip(map_obj_y, top_left[1]+5, bottom_right[1]-5)

            # Enhanced Blips
            cv2.circle(frame, (map_obj_x, map_obj_y), 5, color, -1)
            cv2.circle(frame, (map_obj_x, map_obj_y), 8, color, 1)
            if is_relevant:
                cv2.circle(frame, (map_obj_x, map_obj_y), 12, (0, 0, 255), 1)

        return frame

    def draw_hud(self, frame, mode_text, lane_type, safety_score, live_stats):
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
        
        # Visualize Lanes (DISABLED by user request)
        # for line in detected_lines:
        #     x1, y1, x2, y2 = line
        #     cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        frame, mode_text = self.apply_night_vision(frame, vehicle_boxes_only)
        frame = self.draw_augmented_grid(frame)

        critical_obj_center = None
        min_ttc_val = 100.0

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
                        if tl_text: cv2.putText(frame, tl_text, (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        # No extra info for traffic light

                    raw_dist = self.estimate_distance_geometric(y2, h)
                    stable_dist, velocity, accel = raw_dist, 0, 0
                    is_cut_in = False

                    if track_id != -1:
                        self.last_seen[track_id] = current_time
                        if track_id not in self.tracked_ids: self.tracked_ids.add(track_id)

                        stable_dist, velocity, accel = self.update_kalman(track_id, raw_dist, dt)
                        is_cut_in, lat_vel = self.detect_cut_in(track_id, (x1+x2)/2, dt, w)
                        self.update_trajectory(track_id, ((x1+x2)//2, (y1+y2)//2))

                    # Determine relevance early for minimap
                    in_ego_lane = self.is_in_ego_lane([x1, y1, x2, y2], w, h)
                    is_relevant = False
                    if self.track_lifetime.get(track_id, 0) > 5 or is_cut_in:
                        is_relevant = in_ego_lane or is_cut_in

                    minimap_objects.append([x1, y1, x2, y2, color, stable_dist, is_relevant])

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

                            if is_relevant:
                                info_text = f"{stable_dist:.0f}m"
                                if abs(speed_kmh) > 5: info_text += f" {int(abs(speed_kmh))}kmh"

                                text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                                cv2.rectangle(frame, (x1, y1-20), (x1 + text_size[0] + 10, y1), color, -1)
                                cv2.putText(frame, info_text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
                            else:
                                cv2.putText(frame, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                            # Alerts
                            if is_relevant and label in ['car', 'truck', 'bus'] and track_id != -1:
                                sig = self.detect_vehicle_signals(frame, [x1, y1, x2, y2], track_id)
                                if sig:
                                    sig_color = (0, 0, 255) if "BRAKE" in sig else (0, 165, 255)
                                    cv2.putText(frame, sig, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sig_color, 2)
                                    if "BRAKE" in sig: is_braking = True

                            if is_cut_in: cv2.putText(frame, "CUT-IN", (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                            if is_relevant and is_braking:
                                cv2.putText(frame, "BRAKING", (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                color = (0, 0, 255)

                            if is_relevant and ttc < 3.0:
                                if risk_prob > 50:
                                    cv2.putText(frame, f"RISK {risk_prob}%", (x1, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                    color = (0, 0, 255)
                                if ttc < min_ttc_val:
                                    min_ttc_val = ttc
                                    critical_obj_center = ((x1+x2)//2, y2)

                            if is_relevant and stable_dist < stop_dist_req and stable_dist < 50:
                                 cv2.putText(frame, "CLOSE", (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                            width = x2 - x1
                            if label in ['person', 'bicycle', 'motorcycle', 'dog', 'cow'] and stable_dist < 20:
                                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                                radius = max(width, y2-y1) // 2 + 10
                                if int(time.time() * 5) % 2 == 0:
                                    cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), 2)
                                    cv2.putText(frame, "WATCH OUT", (center_x-40, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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

        frame = self.draw_minimap(frame, minimap_objects)
        # Pass Live Stats to HUD
        frame = self.draw_hud(frame, mode_text, lane_type, safety_score, live_stats)

        return frame

# 5. EXECUTION
# -----------------------------------
print("Initializing ADAS V34.0 (LOCAL VERSION)...")
processor = ADASProcessor()

# Initialize Webcam
print(f"Opening camera input: {VIDEO_SOURCE}")
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Set resolution to match the original script's optimization (640x360)
# Only applies if using a webcam (index)
if isinstance(VIDEO_SOURCE, int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print(f"Error: Could not open video source '{VIDEO_SOURCE}'.")
    print("Please check your camera connection or video file path.")
    sys.exit()

print("Camera opened successfully. Starting Dashboard...")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    try:
        # Process frame
        final_frame = processor.process_frame(frame)

        # Display result
        cv2.imshow('ADAS V34.0', final_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error: {e}")
        break

cap.release()
cv2.destroyAllWindows()