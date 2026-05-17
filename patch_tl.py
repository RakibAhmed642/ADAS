import os

file_path = "e:/DTC/new.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

old_tl_logic = """    def analyze_traffic_light(self, frame, box):
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return (0, 255, 255), "LIGHT"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Red
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Green
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # Yellow
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        r_sum = cv2.countNonZero(mask_red)
        g_sum = cv2.countNonZero(mask_green)
        y_sum = cv2.countNonZero(mask_yellow)

        maximum = max(r_sum, g_sum, y_sum)
        if maximum < 10: return (200, 200, 200), "OFF" # Not lit enough
        
        if maximum == r_sum: return (0, 0, 255), "RED"
        if maximum == g_sum: return (0, 255, 0), "GREEN"
        if maximum == y_sum: return (0, 255, 255), "YELLOW"
        
        return (0, 255, 255), "LIGHT\""""

new_tl_logic = """    def analyze_traffic_light(self, frame, box):
        x1, y1, x2, y2 = box
        
        # Sanity check for minimum dimension
        if (x2 - x1) < 5 or (y2 - y1) < 10:
            return (0, 255, 255), "LIGHT"
            
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return (0, 255, 255), "LIGHT"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        
        # 1. Advanced Brightness Masking (Eliminate unlit parts of the light housing)
        # Using L-channel from LAB space (perceptual brightness)
        l_channel = lab[:, :, 0]
        # Calculate dynamic threshold based on the ROI brightness
        mean_l = np.mean(l_channel)
        # Only analyze pixels that are brighter than average + standard buffer, minimum of 130
        bright_mask = (l_channel > max(130, mean_l + 20)).astype(np.uint8) * 255
        
        # 2. Strict HSV Spectrum Ranges for LED Traffic Lights
        # Red
        mask_red1 = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([12, 255, 255]))
        mask_red2 = cv2.inRange(hsv, np.array([160, 120, 120]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # Green (Traffic green is often cyan-tinted LED)
        mask_green = cv2.inRange(hsv, np.array([45, 100, 100]), np.array([95, 255, 255]))
        
        # Yellow
        mask_yellow = cv2.inRange(hsv, np.array([15, 150, 150]), np.array([35, 255, 255]))
        
        # Apply Brightness Filter
        mask_red = cv2.bitwise_and(mask_red, bright_mask)
        mask_green = cv2.bitwise_and(mask_green, bright_mask)
        mask_yellow = cv2.bitwise_and(mask_yellow, bright_mask)

        # 3. Spatial Zoning Logic (Top = Red, Middle = Yellow, Bottom = Green)
        h_roi = roi.shape[0]
        third = h_roi // 3
        
        # Divide masks into spatial 1/3rds
        # Red should predominantly be in the top 50%
        red_top = cv2.countNonZero(mask_red[0:h_roi//2, :])
        red_bot = cv2.countNonZero(mask_red[h_roi//2:, :])
        
        # Green should predominantly be in the bottom 50%
        green_top = cv2.countNonZero(mask_green[0:h_roi//2, :])
        green_bot = cv2.countNonZero(mask_green[h_roi//2:, :])
        
        # Yellow usually in the middle 3rd
        yellow_mid = cv2.countNonZero(mask_yellow[third:2*third, :])
        
        # Calculate total weighted scores
        # Penalize if color is in the wrong zoning area
        r_score = red_top - (red_bot * 0.5)
        g_score = green_bot - (green_top * 0.5) 
        y_score = yellow_mid
        
        maximum = max(r_score, g_score, y_score)
        
        # Dynamic base threshold required (pixels) based on box volume
        min_pixels_required = (h_roi * (x2 - x1)) * 0.03 # 3% of the bounding box must be lit
        
        if maximum < min(10, min_pixels_required): 
            return (200, 200, 200), "OFF" # Not lit enough physically
            
        # 4. Final Verdict Mapping
        if maximum == r_score: 
            return (0, 0, 255), "RED"    # Red physical light (BGR: 0,0,255)
        if maximum == g_score: 
            return (0, 255, 0), "GREEN"  # Green physical light
        if maximum == y_score: 
            return (0, 255, 255), "YELLOW"
        
        return (0, 255, 255), "LIGHT" """

if old_tl_logic in content:
    content = content.replace(old_tl_logic, new_tl_logic)
else:
    print("WARNING: Could not perfectly match old ALGORITHM string.")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied for Advanced Traffic Light Detection.")
