import os

file_path = "e:/DTC/new.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Imports and ThreadedVideoStream
if "import threading" not in content:
    content = content.replace(
        "from datetime import datetime",
        """from datetime import datetime
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
"""
    )
    
# 2. Init Method variables
if "self.pothole_alert = False" not in content:
    content = content.replace(
        "        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))",
        """        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.pothole_alert = False
        self.ldw_warning = None"""
    )

# 3. Environment Analysis (Potholes)
content = content.replace(
    """        road_roi = gray[int(h*0.7):h, int(w*0.3):int(w*0.7)]
        if road_roi.size > 0:
            mean, std_dev = cv2.meanStdDev(road_roi)
            if std_dev[0][0] > 50: self.road_status = "BUMPY"
            else: self.road_status = "SMOOTH\"""",
    """        road_roi = gray[int(h*0.7):h, int(w*0.3):int(w*0.7)]
        self.pothole_alert = False
        if road_roi.size > 0:
            mean, std_dev = cv2.meanStdDev(road_roi)
            if std_dev[0][0] > 75: 
                self.road_status = "SEVERE ROAD"
                self.pothole_alert = True
            elif std_dev[0][0] > 50: self.road_status = "BUMPY"
            else: self.road_status = "SMOOTH\""""
)

# 4. Radar Sweep
content = content.replace(
    """        # Field of View lines
        cv2.line(frame, (center_x, bottom_y), (top_left[0], top_left[1]), (50, 80, 90), 1)
        cv2.line(frame, (center_x, bottom_y), (bottom_right[0], top_left[1]), (50, 80, 90), 1)

        max_dist = 100.0 # Extended range""",
    """        # Field of View lines
        cv2.line(frame, (center_x, bottom_y), (top_left[0], top_left[1]), (50, 80, 90), 1)
        cv2.line(frame, (center_x, bottom_y), (bottom_right[0], top_left[1]), (50, 80, 90), 1)

        # Radar Sweep Animation (Aesthetic Enhancement)
        sweep_angle = (time.time() * 60) % 180 # 0 to 180 degrees
        sweep_rad = np.radians(180 - sweep_angle) # Sweep from left to right
        sweep_x = int(center_x + map_w * 0.8 * np.cos(sweep_rad))
        sweep_y = int(bottom_y - map_w * 0.8 * np.sin(sweep_rad))
        
        if sweep_y < bottom_y:
            sweep_pts = np.array([[center_x, bottom_y], [sweep_x, sweep_y], [int(center_x + map_w * 0.8 * np.cos(sweep_rad - 0.2)), int(bottom_y - map_w * 0.8 * np.sin(sweep_rad - 0.2))]])
            overlay_radar = frame.copy()
            cv2.fillPoly(overlay_radar, [sweep_pts], (0, 255, 0))
            frame = cv2.addWeighted(overlay_radar, 0.15, frame, 0.85, 0)
            cv2.line(frame, (center_x, bottom_y), (sweep_x, sweep_y), (0, 255, 0), 1)

        max_dist = 100.0 # Extended range"""
)

# 5. HUD active signs drawing
content = content.replace(
    """        # --- DRAW ACTIVE ROAD SIGNS ON HUD ---
        if active_signs:
            sign_y = panel_y + panel_h + 20""",
    """        # --- DRAW ACTIVE ROAD SIGNS & ALERTS ON HUD ---
        if getattr(self, 'pothole_alert', False):
            active_signs.append(("POTHOLE DETECTED!", (0, 0, 255)))
        if getattr(self, 'ldw_warning', None):
            active_signs.append((self.ldw_warning, (0, 0, 255)))
            
        if active_signs:
            sign_y = panel_y + panel_h + 20"""
)


# 6. Process frame loop LDW & Traffic Signs
content = content.replace(
    """        # Lane Detection
        detected_lines = self.detect_lane_lines(frame)
        lane_type = self.estimate_lane_count(vehicle_boxes_only, w, detected_lines)
        
        # Visualize Lanes (DISABLED by user request)
        # for line in detected_lines:
        #     x1, y1, x2, y2 = line
        #     cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        frame, mode_text = self.apply_night_vision(frame, vehicle_boxes_only)""",
    """        # Lane Detection
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

        frame, mode_text = self.apply_night_vision(frame, vehicle_boxes_only)"""
)


# 7. Trajectory path and Multi-tier FCW + Cross traffic
content = content.replace(
    """                    if track_id != -1:
                        self.last_seen[track_id] = current_time
                        if track_id not in self.tracked_ids: self.tracked_ids.add(track_id)

                        stable_dist, velocity, accel = self.update_kalman(track_id, raw_dist, dt)
                        is_cut_in, lat_vel, intent_text = self.detect_lateral_intent(track_id, (x1+x2)/2, dt, w)
                        self.update_trajectory(track_id, ((x1+x2)//2, (y1+y2)//2))""",
    """                    if track_id != -1:
                        self.last_seen[track_id] = current_time
                        if track_id not in self.tracked_ids: self.tracked_ids.add(track_id)

                        stable_dist, velocity, accel = self.update_kalman(track_id, raw_dist, dt)
                        is_cut_in, lat_vel, intent_text = self.detect_lateral_intent(track_id, (x1+x2)/2, dt, w)
                        self.update_trajectory(track_id, ((x1+x2)//2, (y1+y2)//2))
                        
                        # --- DRAW PREDICTIVE TRAJECTORY TAIL ---
                        if len(self.object_paths[track_id]) > 3:
                            pts = list(self.object_paths[track_id])
                            for i in range(1, len(pts)):
                                thickness = int(np.sqrt(i) * 1.5)
                                # Fading color effect
                                tail_color = (int(color[0]*i/len(pts)), int(color[1]*i/len(pts)), int(color[2]*i/len(pts)))
                                cv2.line(frame, pts[i-1], pts[i], tail_color, thickness)"""
)

content = content.replace(
    """                            # Advanced adjustment to risk based on impact severity
                            risk_prob = min(100, int(risk_prob + impact_severity * 10))
                            max_risk_prob = max(max_risk_prob, risk_prob)

                            # Initialize label_text here to fix NameError
                            label_text = f"{label[:3].upper()}\"

                            if is_relevant:""",
    """                            # Advanced adjustment to risk based on impact severity
                            risk_prob = min(100, int(risk_prob + impact_severity * 10))
                            max_risk_prob = max(max_risk_prob, risk_prob)

                            # Initialize label_text here to fix NameError
                            label_text = f"{label[:3].upper()}\"

                            # --- CROSS TRAFFIC ALERT ---
                            is_cross_traffic = False
                            if label in ['car', 'truck', 'bus'] and abs(lat_vel) > 30 and ttc < 4.0 and in_ego_lane == False:
                                is_cross_traffic = True
                                cv2.putText(frame, "CROSS TRAFFIC!", (w//2 - 100, h//2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                            # --- MULTI-TIER FCW OVERLAY ---
                            if is_relevant and ttc < 5.0 and not is_cut_in:
                                # Watch -> Warning -> Critical
                                box_color = (0, 255, 255) # Yellow
                                if ttc < 3.0: box_color = (0, 165, 255) # Orange
                                if ttc < 1.5: box_color = (0, 0, 255) # Red Critical
                                # Pulsing effect
                                if ttc < 3.0 and int(time.time() * 10) % 2 == 0:
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)

                            if is_relevant:"""
)


# 8. Main execution
content = content.replace(
    """# 5. EXECUTION
# -----------------------------------
print("Initializing ADAS V34.0 (LOCAL VERSION)...")
processor = ADASProcessor()

# Initialize Webcam
print(f"Opening camera input: {VIDEO_SOURCE}\")
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
        break""",
    """# 5. EXECUTION
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
        continue # Wait for next threaded frame"""
)

content = content.replace(
    """# Display result
        cv2.imshow('ADAS V34.0', final_frame)""",
    """# Display result
        cv2.imshow('ADAS V35.0 PRO', final_frame)"""
)


with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied successfully.")
