# ==========================================
# REAL-TIME ADAS SYSTEM V34.5 (AUTO-FIX + LOCAL + GPU FORCE)
# Modified for Windows/Local PC Execution
# Features: Live Traffic Stats, Red Strip Blind Spot, Traffic Light Filter
# ==========================================

import os
import cv2
import numpy as np
import time
import torch
import sys
from collections import deque
from datetime import datetime

# 1. SETUP & STRICT GPU CHECK
# -----------------------------------
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing Ultralytics (YOLOv8)... Please wait.")
    os.system('pip install ultralytics')
    from ultralytics import YOLO

# FORCE GPU CHECK & DIAGNOSTICS
print("\n" + "="*50)
print("🔍 SYSTEM CHECK...")
print(f"   PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"✅ SUCCESS: GPU DETECTED -> {torch.cuda.get_device_name(0)}")
    try:
        print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except:
        pass
    device = 'cuda'
else:
    print("❌ ERROR: GPU NOT DETECTED BY PYTORCH!")
    print("   Your PC likely has a GPU, but the installed PyTorch is 'CPU-only'.")
    
    # Check if user has CPU version installed
    if 'cpu' in torch.__version__:
        print("\n⚠️ DIAGNOSIS: You have the CPU version of PyTorch installed.")
        print("   The script can try to fix this automatically.")
        
        # AUTO-FIX PROMPT
        print("\n✨ AUTO-FIX AVAILABLE ✨")
        choice = input("👉 Do you want to uninstall the CPU version and install the GPU version now? (y/n): ")
        
        if choice.lower() == 'y':
            print("\n⏳ Step 1/2: Uninstalling current 'cpu' torch (this might take a moment)...")
            os.system('pip uninstall torch torchvision torchaudio -y')
            
            print("\n⏳ Step 2/2: Downloading & Installing GPU-enabled PyTorch (CUDA 12.1)...")
            print("   (This is a large download ~2.5GB, please wait...)")
            os.system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
            
            print("\n✅ Installation complete!")
            print("👉 PLEASE CLOSE THIS WINDOW AND RE-RUN THE SCRIPT to use the GPU.")
            print("="*50)
            sys.exit()
        else:
            print("\n⚠️ You chose not to fix it. Running on CPU (Will be slow)...")
    else:
        print("\n⚠️ DIAGNOSIS: CUDA Drivers might be missing.")
        print("   Please install NVIDIA CUDA Toolkit 12.x from NVIDIA website.")

    device = 'cpu'

print("="*50 + "\n")

print("Loading MAX POWER Model (Extra Large) for Ultimate Accuracy...")
# Using 'yolov8x.pt' (Extra Large)
model = YOLO('yolov8x.pt') 
if device == 'cuda':
    model.to('cuda')
    # GPU WARMUP
    print("🔥 Warming up GPU...")
    dummy_input = np.zeros((360, 640, 3), dtype=np.uint8)
    for _ in range(3):
        # Explicitly force device=0
        model.predict(dummy_input, verbose=False, half=True, device=0)
    print("🚀 GPU Ready & Warmed Up!")

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
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
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

        # Colors
        self.colors = {
            'car': (255, 255, 0), 'person': (0, 165, 255),
            'truck': (255, 0, 255), 'bus': (0, 128, 255),
            'motorcycle': (50, 205, 50), 'bicycle': (255, 191, 0),
            'traffic light': (200, 200, 200), 'stop sign': (0, 0, 200),
            'dog': (180, 105, 255), 'cow': (180, 105, 255),
            'horse': (180, 105, 255), 'sheep': (180, 105, 255),
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

        if sharpness < 80:
            self.weather_status = "RAIN"
            self.friction_coef = 0.4
        elif sharpness < 150:
            self.weather_status = "CLOUD"
            self.friction_coef = 0.6
        else:
            self.weather_status = "CLEAR"
            self.friction_coef = 0.8

        if vehicle_count > 8: self.traffic_status = "JAM"
        elif vehicle_count > 3: self.traffic_status = "BUSY"
        else: self.traffic_status = "FREE"

        h, w = frame.shape[:2]
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
        return ttc, headway, is_braking

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

    def estimate_lane_count(self, boxes, w):
        if not boxes: return "1-LANE"
        x_centers = sorted([(box[0] + box[2]) / 2 for box in boxes])
        clusters = 0
        if x_centers:
            clusters = 1
            last_x = x_centers[0]
            lane_threshold = w * 0.15
            for i in range(1, len(x_centers)):
                if x_centers[i] - last_x > lane_threshold:
                    clusters += 1
                    last_x = x_centers[i]
        self.lane_history.append(clusters)
        avg_lanes = sum(self.lane_history) / len(self.lane_history)
        if avg_lanes <= 1.5: return "1-LN"
        elif avg_lanes <= 2.5: return "2-LN"
        else: return "3+LN"

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

    def detect_vehicle_signals(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        if w < 40 or h < 40: return None

        lights_y1 = y1 + int(h * 0.3)
        lights_y2 = y1 + int(h * 0.8)
        if lights_y1 >= lights_y2: return None

        roi = frame[lights_y1:lights_y2, x1:x2]
        if roi.size == 0: return None

        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        roi_h, roi_w = roi.shape[:2]
        w_split = int(roi_w * 0.25)

        left_l = l_channel[:, :w_split]
        right_l = l_channel[:, roi_w - w_split:]

        left_a = a_channel[:, :w_split]
        right_a = a_channel[:, roi_w - w_split:]

        red_thresh = 150
        bright_thresh = 130

        left_red_mask = (left_a > red_thresh) & (left_l > bright_thresh)
        right_red_mask = (right_a > red_thresh) & (right_l > bright_thresh)

        l_red_px = cv2.countNonZero(left_red_mask.astype(np.uint8))
        r_red_px = cv2.countNonZero(right_red_mask.astype(np.uint8))

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        amber_lower = np.array([15, 150, 150])
        amber_upper = np.array([35, 255, 255])

        left_hsv = hsv_roi[:, :w_split]
        right_hsv = hsv_roi[:, roi_w - w_split:]

        l_amber_px = cv2.countNonZero(cv2.inRange(left_hsv, amber_lower, amber_upper))
        r_amber_px = cv2.countNonZero(cv2.inRange(right_hsv, amber_lower, amber_upper))

        threshold = (w_split * roi_h) * 0.05

        if l_red_px > threshold and r_red_px > threshold: return "BRAKE"
        if l_amber_px > threshold and l_amber_px > r_amber_px * 2: return "LEFT TURN"
        if r_amber_px > threshold and r_amber_px > l_amber_px * 2: return "RIGHT TURN"
        if l_red_px > threshold and l_red_px > r_red_px * 5: return "LEFT TURN (RED)"
        if r_red_px > threshold and r_red_px > l_red_px * 5: return "RIGHT TURN (RED)"
        return None

    def draw_minimap(self, frame, objects):
        h, w = frame.shape[:2]
        map_w = 160
        map_h = 160

        overlay = frame.copy()
        top_left = (20, h - map_h - 20)
        bottom_right = (20 + map_w, h - 20)
        cv2.rectangle(overlay, top_left, bottom_right, (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        cv2.rectangle(frame, top_left, bottom_right, (100, 100, 100), 1)

        center_x = top_left[0] + map_w // 2
        bottom_y = bottom_right[1] - 10
        cv2.arrowedLine(frame, (center_x, bottom_y), (center_x, bottom_y - 15), (0, 255, 255), 3, tipLength=0.5)

        max_dist = 80.0
        scale = (map_h - 20) / max_dist

        for d in [20, 40, 60]:
            y_ring = bottom_y - int(d * scale)
            if y_ring > top_left[1]:
                cv2.ellipse(frame, (center_x, bottom_y), (map_w//2, int(d*scale)), 0, 180, 360, (50, 50, 50), 1)
                cv2.putText(frame, f"{d}m", (center_x+2, y_ring), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

        for obj in objects:
            x1, y1, x2, y2, color, dist_m = obj
            cx, cy = (x1 + x2) // 2, y2
            rel_x = (cx - (w/2)) / (w/2)
            map_x_offset = int(rel_x * (map_w / 2))
            map_obj_x = center_x + map_x_offset
            map_obj_y = bottom_y - int(dist_m * scale)

            map_obj_x = np.clip(map_obj_x, top_left[0]+5, bottom_right[0]-5)
            map_obj_y = np.clip(map_obj_y, top_left[1]+5, bottom_right[1]-5)

            cv2.circle(frame, (map_obj_x, map_obj_y), 4, color, -1)

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
        panel_h = 160
        panel_x = w - panel_w - 10
        panel_y = 60

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 1)

        self.draw_text_with_outline(frame, "TRAFFIC", (panel_x + 35, panel_y + 20), 0.5, (200, 200, 200))
        cv2.line(frame, (panel_x + 10, panel_y + 25), (panel_x + panel_w - 10, panel_y + 25), (100, 100, 100), 1)

        y_off = panel_y + 50
        icons = {'car': 'CAR', 'person': 'PED', 'truck': 'TRK', 'bus': 'BUS', 'motorcycle': 'MOTO'}
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
        
        # Calculate FPS
        if dt > 0:
            self.fps = 1.0 / dt

        current_vehicle_count = 0
        h, w = frame.shape[:2]

        horizon_level = int(h * 0.45)
        road_contour = np.array([[0, h], [w, h], [int(w * 0.70), horizon_level], [int(w * 0.30), horizon_level]], dtype=np.int32)

        t0 = time.time()
        # Use device argument dynamically
        global device
        # Ensure device=0 is passed if using cuda to prevent CPU fallback
        results = model.track(frame, persist=True, conf=0.25, iou=0.5, agnostic_nms=True, verbose=False, device=0 if device == 'cuda' else 'cpu', half=(device == 'cuda'))
        t1 = time.time()
        self.inference_time = (t1 - t0) * 1000

        detected_boxes = []
        minimap_objects = []
        vehicle_boxes_only = []
        max_risk_prob = 0

        # New: Live Frame Stats Dictionary
        live_stats = {k: 0 for k in ['car', 'person', 'truck', 'bus', 'motorcycle']}

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in ['car', 'truck', 'bus', 'motorcycle']:
                    vehicle_boxes_only.append([x1, y1, x2, y2])
                    current_vehicle_count += 1

                # Update live stats
                if label in live_stats:
                    live_stats[label] += 1

        if self.frame_count % 30 == 0:
            self.clean_memory(current_time)
            self.analyze_environment(frame, current_vehicle_count)
        
        self.frame_count += 1

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

                    minimap_objects.append([x1, y1, x2, y2, color, stable_dist])

                    if self.track_lifetime.get(track_id, 0) > 5 or is_cut_in:

                        l_len = int((x2-x1) * 0.2)
                        cv2.line(frame, (x1, y1), (x1+l_len, y1), color, 2)
                        cv2.line(frame, (x1, y1), (x1, y1+l_len), color, 2)
                        cv2.line(frame, (x2, y2), (x2-l_len, y2), color, 2)
                        cv2.line(frame, (x2, y2), (x2, y2-l_len), color, 2)

                        # --- FIX: Skip Physics/Info for Traffic Light ---
                        if label != 'traffic light':
                            speed_kmh = velocity * 3.6
                            ttc, headway, is_braking = self.calculate_physics_metrics(velocity, stable_dist, accel)
                            stop_dist_req = self.calculate_stopping_distance(abs(speed_kmh))
                            risk_prob = self.calculate_collision_risk(ttc)
                            max_risk_prob = max(max_risk_prob, risk_prob)

                            # Initialize label_text here to fix NameError
                            label_text = f"{label[:3].upper()}"

                            info_text = f"{stable_dist:.0f}m"
                            if abs(speed_kmh) > 5: info_text += f" {int(abs(speed_kmh))}kmh"

                            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                            cv2.rectangle(frame, (x1, y1-20), (x1 + text_size[0] + 10, y1), color, -1)
                            cv2.putText(frame, info_text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

                            # Alerts
                            if label in ['car', 'truck', 'bus']:
                                sig = self.detect_vehicle_signals(frame, [x1, y1, x2, y2])
                                if sig:
                                    sig_color = (0, 0, 255) if "BRAKE" in sig else (0, 165, 255)
                                    cv2.putText(frame, sig, (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sig_color, 2)
                                    if "BRAKE" in sig: is_braking = True

                            if is_cut_in: cv2.putText(frame, "CUT-IN", (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                            if is_braking:
                                cv2.putText(frame, "BRAKING", (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                color = (0, 0, 255)

                            if ttc < 3.0:
                                if risk_prob > 50:
                                    cv2.putText(frame, f"RISK {risk_prob}%", (x1, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                    color = (0, 0, 255)
                                if ttc < min_ttc_val:
                                    min_ttc_val = ttc
                                    critical_obj_center = ((x1+x2)//2, y2)

                            if stable_dist < stop_dist_req and stable_dist < 50:
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
        lane_type = self.estimate_lane_count(detected_boxes, w)
        safety_score = self.calculate_safety_score(min_ttc_val, self.weather_status, self.road_status)

        frame = self.draw_minimap(frame, minimap_objects)
        # Pass Live Stats to HUD
        frame = self.draw_hud(frame, mode_text, lane_type, safety_score, live_stats)

        return frame

# 5. EXECUTION (LOCAL + CAMERA SELECT)
# -----------------------------------
def select_camera_source():
    """Scans for available cameras and lets the user choose."""
    print("\n🔍 Scanning for available cameras (Indices 0-5)...")
    available_cams = []
    
    # Check first 5 indices
    for i in range(5):
        try:
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                ret, _ = temp_cap.read()
                if ret:
                    available_cams.append(i)
                    print(f"   [✓] Camera Index {i}: Available")
                temp_cap.release()
        except:
            pass
    
    if not available_cams:
        print("❌ No valid cameras found!")
        print("   Using default Index 0 hoping for the best...")
        return 0
    
    if len(available_cams) == 1:
        cam_idx = available_cams[0]
        print(f"✅ Only one camera found. Auto-selecting Index {cam_idx}")
        return cam_idx
        
    while True:
        try:
            print("\n📸 Available Cameras:", available_cams)
            selection = input("👉 Enter Camera Index to use: ")
            idx = int(selection)
            if idx in available_cams:
                return idx
            else:
                print("❌ Invalid selection. Please enter one of the numbers listed above.")
        except ValueError:
            print("❌ Please enter a numeric value.")

if __name__ == "__main__":
    print("Initializing ADAS V34.0 Local Version...")
    
    # 1. Select Camera
    selected_cam_index = select_camera_source()
    
    processor = ADASProcessor()
    
    # 2. Open Selected Camera
    cap = cv2.VideoCapture(selected_cam_index)
    
    # Set Resolution for better performance/quality balance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 440)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 260)

    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {selected_cam_index}.")
        exit()

    print(f"🎥 Camera {selected_cam_index} Started. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Process the frame
        try:
            final_frame = processor.process_frame(frame)
            
            # Show output
            cv2.imshow("ADAS System V34 (Local)", final_frame)
        except Exception as e:
            print(f"Error processing frame: {e}")

        # Exit on 'Q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()