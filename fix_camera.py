import os

file_path = "e:/DTC/new.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix camera freezup / 100% CPU lock in ThreadedVideoStream loop
if "time.sleep(0.005) # Prevent 100% CPU" not in content:
    content = content.replace(
        """    if not ret or frame is None:
        continue # Wait for next threaded frame""",
        """    if not ret or frame is None:
        time.sleep(0.005) # Prevent 100% CPU lock when queue is empty!
        continue # Wait for next threaded frame"""
    )
    
# Replace detect_vehicle_signals perfectly
old_detect = """    def detect_vehicle_signals(self, frame, box, track_id):
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
        if left_count > len(history) * 0.3 and right_count < 2:
            return "SIGNAL LEFT"
            
        if right_count > len(history) * 0.3 and left_count < 2:
            return "SIGNAL RIGHT"

        return None"""

new_detect = """    def detect_vehicle_signals(self, frame, box, track_id):
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        if w < 30 or h < 30: return "STRAIGHT"

        # ROI: Focus on upper-middle rear (tail lights)
        lights_y1 = y1 + int(h * 0.35)
        lights_y2 = y1 + int(h * 0.65)
        if lights_y1 >= lights_y2: return "STRAIGHT"

        roi = frame[lights_y1:lights_y2, x1:x2]
        if roi.size == 0: return "STRAIGHT"

        # Split into Left and Right regions for turn signals
        roi_h, roi_w = roi.shape[:2]
        w_split = int(roi_w * 0.30)
        
        left_roi = roi[:, :w_split]
        right_roi = roi[:, roi_w - w_split:]

        # Analysis Function - Highly precise color and brightness detection
        def get_light_intensity(img_part):
            hsv = cv2.cvtColor(img_part, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img_part, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # Vibrant Red/Orange/Yellow masks (Broader to catch LED and Halogen)
            mask1 = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([15, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([160, 70, 70]), np.array([180, 255, 255]))
            mask3 = cv2.inRange(hsv, np.array([15, 70, 70]), np.array([35, 255, 255]))
            
            color_mask = mask1 | mask2 | mask3
            bright_mask = (l_channel > 120)
            
            final_mask = color_mask & bright_mask
            return cv2.countNonZero(final_mask.astype(np.uint8))

        l_val = get_light_intensity(left_roi)
        r_val = get_light_intensity(right_roi)
        
        # Check Saturation to filter out white headlights or street reflections
        def check_saturation(img_part):
            hsv = cv2.cvtColor(img_part, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            bright_mask = (v > 150).astype(np.uint8)
            if cv2.countNonZero(bright_mask) > 5:
                # Return 0 to 255
                return np.mean(s[bright_mask])
            return 255 
            
        l_sat = check_saturation(left_roi)
        r_sat = check_saturation(right_roi)
        
        # If very white, ignore
        if l_sat < 40 and r_sat < 40:
             return "STRAIGHT"

        # Smarter dynamic pixel threshold based on box size
        threshold = max(3, (w_split * roi_h) * 0.015) 

        current_state = 'OFF'
        # Detect Brake (Both sides are glowing red and pass threshold)
        if l_val > threshold and r_val > threshold:
             current_state = 'BRAKE'
        # Detect Turn signal (Extreme imbalance leaning towards one side)
        elif l_val > threshold and l_val > r_val * 2.0:
            current_state = 'LEFT'
        elif r_val > threshold and r_val > l_val * 2.0:
            current_state = 'RIGHT'
            
        # --- Deep Temporal Smoothing Engine (Debounce & Pattern Detection) ---
        if track_id not in self.signal_history:
            self.signal_history[track_id] = deque(maxlen=20) # Track 20 frames for perfect stability
            
        self.signal_history[track_id].append(current_state)
        history = list(self.signal_history[track_id])
        
        brake_count = history.count('BRAKE')
        left_count = history.count('LEFT')
        right_count = history.count('RIGHT')
        
        # Mathematical ratio checks
        total_frames = len(history)
        
        # Very reliable hard braking
        if brake_count > total_frames * 0.35:
            return "BRAKE"
            
        # Turn signal rhythmic tracking
        # If predominately blinking LEFT and almost ZERO RIGHT counts
        if left_count > total_frames * 0.20 and right_count <= 2:
            return "SIGNAL LEFT"
            
        if right_count > total_frames * 0.20 and left_count <= 2:
            return "SIGNAL RIGHT"

        return "STRAIGHT" """

if old_detect in content:
    content = content.replace(old_detect, new_detect)
else:
    print("Warning: old detect_vehicle_signals not found exactly.")
    # Fallback to replace it using a script approach if needed, but strings are exact.

# Now update the UI drawing block
old_ui = """                                # Do NOT check for rear signals if the car is facing us (ONCOMING)
                                sig = None
                                if vehicle_direction != "ONCOMING":
                                    sig = self.detect_vehicle_signals(frame, [x1, y1, x2, y2], track_id)
                                
                                final_intent = None
                                sig_color = (0, 165, 255) # Default Warning Orange
                                
                                if sig == "BRAKE" or is_braking:
                                    final_intent = "HARD BRAKING"
                                    sig_color = (0, 0, 255)
                                    is_braking = True
                                elif sig == "SIGNAL LEFT" and intent_text == "SHIFT LEFT":
                                    final_intent = "LANE CHANGE: LEFT <-"
                                elif sig == "SIGNAL RIGHT" and intent_text == "SHIFT RIGHT":
                                    final_intent = "LANE CHANGE: RIGHT ->"
                                elif sig == "SIGNAL LEFT":
                                    final_intent = "PREPARING LEFT <-"
                                elif sig == "SIGNAL RIGHT":
                                    final_intent = "PREPARING RIGHT ->"
                                elif intent_text != "STRAIGHT":
                                    # Only physical drift 
                                    final_intent = f"DRIFTING: {intent_text}"
                                    sig_color = (150, 150, 150)"""

new_ui = """                                # Superior Intent & Light State Interpreter
                                sig = "STRAIGHT"
                                if vehicle_direction != "ONCOMING":
                                    sig_result = self.detect_vehicle_signals(frame, [x1, y1, x2, y2], track_id)
                                    if sig_result: sig = sig_result
                                
                                final_intent = None
                                sig_color = (0, 255, 0) # Base Safe Green
                                
                                if sig == "BRAKE" or is_braking:
                                    final_intent = "HARD BRAKING"
                                    sig_color = (0, 0, 255) # Red danger
                                    is_braking = True
                                elif sig == "SIGNAL LEFT" and intent_text == "SHIFT LEFT":
                                    final_intent = "LANE CHANGE: LEFT <-"
                                    sig_color = (0, 165, 255) # Orange warning
                                elif sig == "SIGNAL RIGHT" and intent_text == "SHIFT RIGHT":
                                    final_intent = "LANE CHANGE: RIGHT ->"
                                    sig_color = (0, 165, 255)
                                elif sig == "SIGNAL LEFT":
                                    final_intent = "TURNING: LEFT <-"
                                    sig_color = (0, 165, 255)
                                elif sig == "SIGNAL RIGHT":
                                    final_intent = "TURNING: RIGHT ->"
                                    sig_color = (0, 165, 255)
                                elif sig == "STRAIGHT":
                                    final_intent = "HEADING STRAIGHT"
                                    sig_color = (0, 255, 0) # Green 
                                elif intent_text != "STRAIGHT":
                                    final_intent = f"DRIFTING: {intent_text}"
                                    sig_color = (150, 150, 150)"""

if old_ui in content:
    content = content.replace(old_ui, new_ui)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied.")
