import os

file_path = "e:/DTC/new.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

old_minimap = """    def draw_minimap(self, frame, objects):
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

        max_dist = 100.0 # Extended range
        scale = (map_h - 20) / max_dist

        for d in [25, 50, 75, 100]:
            y_ring = bottom_y - int(d * scale)
            if y_ring > top_left[1]:
                cv2.ellipse(frame, (center_x, bottom_y), (int(d * scale * 1.5), int(d*scale)), 0, 180, 360, (50, 80, 90), 1)
                cv2.putText(frame, f"{d}m", (center_x+2, y_ring-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 200, 200), 1)

        for obj in objects:
            if len(obj) == 8:
                x1, y1, x2, y2, color, dist_m, is_relevant, track_id = obj
            elif len(obj) == 7:
                x1, y1, x2, y2, color, dist_m, is_relevant = obj
                track_id = -1
            else:
                x1, y1, x2, y2, color, dist_m = obj[:6] # Fallback
                is_relevant = False
                track_id = -1
            
            cx, cy = (x1 + x2) // 2, y2
            rel_x = (cx - (w/2)) / (w/2)
            map_x_offset = int(rel_x * (map_w / 1.5)) # Wider spread for realism
            map_obj_x = center_x + map_x_offset
            map_obj_y = bottom_y - int(dist_m * scale)

            map_obj_x = np.clip(map_obj_x, top_left[0]+5, bottom_right[0]-5)
            map_obj_y = np.clip(map_obj_y, top_left[1]+5, bottom_right[1]-5)

            # Enhanced Blips with realistic tracking tails
            if track_id in self.object_paths and len(self.object_paths[track_id]) > 2:
                pts = list(self.object_paths[track_id])
                for i in range(1, len(pts)):
                    px, py = pts[i]
                    rel_px = (px - (w/2)) / (w/2)
                    m_px = center_x + int(rel_px * (map_w / 1.5))
                    
                    # Assuming constant distance for trail simplicity, or better just fading trails
                    pass # Fully 3D trails are complex, sticking to precise glowing blips

            cv2.circle(frame, (map_obj_x, map_obj_y), 4, color, -1)
            cv2.circle(frame, (map_obj_x, map_obj_y), 7, color, 1)
            if is_relevant:
                # Pulsing red ring for relevant threats
                pulse = int(10 + 3 * np.sin(time.time() * 10))
                cv2.circle(frame, (map_obj_x, map_obj_y), pulse, (0, 0, 255), 1)

        return frame"""

new_minimap = """    def draw_minimap(self, frame, objects):
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
        
        # 2. Minimalist FOV Corridor
        cv2.line(map_canvas, (center_x, bottom_y), (0, 0), (70, 75, 75), 1)
        cv2.line(map_canvas, (center_x, bottom_y), (map_w, 0), (70, 75, 75), 1)

        max_dist = 100.0
        scale = bottom_y / max_dist

        # 3. Horizontal Grid Lines (Accurate depth cues instead of arcs spilling out)
        for d in [20, 40, 60, 80]:
            y_d = bottom_y - int(d * scale)
            if y_d > 0:
                # Calculate exact width of FOV lines at this distance
                if bottom_y > 0:
                    span_half_width = int((bottom_y - y_d) * (center_x / bottom_y))
                    cv2.line(map_canvas, (center_x - span_half_width, y_d), (center_x + span_half_width, y_d), (60, 65, 65), 1)
                    cv2.putText(map_canvas, f"{d}m", (center_x + span_half_width + 5, y_d + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 130, 130), 1)

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
            
            # Map object relative to perspective
            cx, cy = (x1 + x2) // 2, y2
            rel_x = (cx - (w/2)) / (w/2)
            
            map_obj_y = bottom_y - int(dist_m * scale)
            # Lateral spread based on FOV at that height
            if bottom_y > 0:
                current_fov_half_width = (bottom_y - map_obj_y) * (center_x / bottom_y)
                # Ensure a minimum spread so objects don't completely overlap on the center line
                current_fov_half_width = max(15, current_fov_half_width * 1.5)
                map_x_offset = int(rel_x * current_fov_half_width)
            else:
                map_x_offset = 0
            
            map_obj_x = center_x + map_x_offset
            
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

        return frame"""

if old_minimap in content:
    content = content.replace(old_minimap, new_minimap)
else:
    print("WARNING: Exact match failed")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Minimap patch applied.")
