import sys
import cv2
import json
import os
import queue
import time
import threading
import requests
import math
from collections import defaultdict, deque
from ultralytics import YOLO
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QLineEdit, QGroupBox, 
                             QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox,
                             QSizePolicy, QScrollArea, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
import paho.mqtt.client as mqtt
from datetime import datetime
import numpy as np

# =========================
# Configuration
# =========================
DEFAULT_VIDEO_SOURCE = r"rtsp://localhost:8554/mystream"
DEFAULT_MODEL_PATH = "yolov8n.pt"  # Using standard YOLO model for person detection
CFG_FILE = "person_roi_config.json"
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "people/counter/control/1"
SERVER_URL = "https://your-server.com/api/people_count"
RECORDINGS_DIR = "person_recordings"  # Directory for saving recordings

# Predefined camera options
CAMERA_OPTIONS = {
    "Camera 1": "rtsp://localhost:8554/mystream",
    "Camera 2": "rtsp://admin:pass%40123@192.168.1.240:554/cam/realmonitor?channel=4&subtype=0",
    "Camera 3": "rtsp://localhost:8554/mystream",
    "Custom RTSP": ""  # Empty string for custom input
}

# =========================
# Frame Capture Thread
# =========================
class FrameCaptureThread(QThread):
    new_frame_signal = pyqtSignal(object)
    stream_status_signal = pyqtSignal(bool, str)
    
    def __init__(self, video_source, frame_queue_size=10, show_connection_status=False):
        super().__init__()
        self.video_source = video_source
        self.frame_queue = queue.Queue(maxsize=frame_queue_size)
        self.running = False
        self.is_stream_active = False
        self.show_connection_status = show_connection_status
        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480
    
    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print(f"Error: Could not open video stream {self.video_source}")
            if self.show_connection_status:
                self.stream_status_signal.emit(False, self.video_source)
            return
        
        # Configure stream properties
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        if self.show_connection_status:
            self.stream_status_signal.emit(True, self.video_source)
        self.is_stream_active = True
        
        # Get initial frame immediately
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            self.new_frame_signal.emit(frame)
            try:
                self.frame_queue.put(frame, timeout=1)
            except queue.Full:
                pass
        
        # Main capture loop
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"Frame grab failed for {self.video_source}, retrying...")
                time.sleep(1)
                continue

            frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            self.new_frame_signal.emit(frame)
            
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame, timeout=1)
            except queue.Full:
                pass
            except Exception as e:
                print(f"Error adding frame to queue: {e}")

        cap.release()
        if self.show_connection_status:
            self.stream_status_signal.emit(False, self.video_source)
        self.is_stream_active = False
    
    def stop(self):
        self.running = False
        self.wait()
    
    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

# =========================
# Worker Thread for Video Processing
# =========================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_count_signal = pyqtSignal(int, int)  # Now emits (in_count, out_count)
    update_info_signal = pyqtSignal(str)
    connection_status_signal = pyqtSignal(bool, str)
    recording_status_signal = pyqtSignal(bool, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.run_flag = True
        self.paused = False
        self.edit_mode = False
        
        # Configuration parameters
        self.video_source = DEFAULT_VIDEO_SOURCE
        self.model_path = DEFAULT_MODEL_PATH
        self.class_filter = [0]  # Filter for person class (COCO dataset class 0)
        self.conf_thresh = 0.25
        self.iou_thresh = 0.45
        self.tracker = "bytetrack.yaml"
        self.axis = 'y'  # Default to vertical counting
        self.forward_positive = True
        
        # ROI and tracking state
        self.roi = None
        self.roi_rotation = 0  # Rotation angle in degrees
        self.obj_state = {}
        self.in_count = 0
        self.out_count = 0
        
        # Mouse interaction state
        self.dragging = False
        self.resizing = False
        self.rotating = False
        self.drag_offset = (0, 0)
        self.resize_corner = None
        self.rotate_handle = None
        self.mouse_down_pt = None
        self.handle_size = 10
        self.rotate_handle_length = 40
        
        # Video capture and model
        self.cap_thread = None
        self.model = None
        
        # Recording state
        self.recording = False
        self.video_writer = None
        self.recording_filename = None
        
        # Create recordings directory if it doesn't exist
        if not os.path.exists(RECORDINGS_DIR):
            os.makedirs(RECORDINGS_DIR)
    
    def start_recording(self):
        """Start video recording"""
        if not self.recording:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.recording_filename = os.path.join(RECORDINGS_DIR, f"person_recording_{timestamp}.mp4")
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.recording_filename, 
                    fourcc, 
                    15.0,
                    (640, 480)
                )
                
                if self.video_writer.isOpened():
                    self.recording = True
                    self.recording_status_signal.emit(True, f"Recording started: {self.recording_filename}")
                else:
                    self.recording_status_signal.emit(False, "Failed to start recording")
                    
            except Exception as e:
                self.recording_status_signal.emit(False, f"Error starting recording: {e}")
    
    def stop_recording(self):
        """Stop video recording"""
        if self.recording and self.video_writer is not None:
            try:
                self.video_writer.release()
                self.video_writer = None
                self.recording = False
                self.recording_status_signal.emit(False, f"Recording saved: {self.recording_filename}")
            except Exception as e:
                self.recording_status_signal.emit(False, f"Error stopping recording: {e}")
    
    def write_frame_to_recording(self, frame):
        """Write annotated frame to recording if recording is active"""
        if self.recording and self.video_writer is not None:
            try:
                self.video_writer.write(frame)
            except Exception as e:
                print(f"Error writing frame to recording: {e}")
    
    def load_config(self):
        if os.path.exists(CFG_FILE):
            try:
                with open(CFG_FILE, "r") as f:
                    data = json.load(f)
                self.roi = self.normalize_roi(data.get("roi"))
                self.roi_rotation = data.get("roi_rotation", 0)
                self.axis = data.get("axis", self.axis)
                self.forward_positive = data.get("forward_positive", self.forward_positive)
                self.in_count = data.get("in_count", 0)
                self.out_count = data.get("out_count", 0)
                self.update_info_signal.emit(f"Loaded configuration from {CFG_FILE}")
            except Exception as e:
                self.update_info_signal.emit(f"Error loading config: {str(e)}")
    
    def save_config(self):
        try:
            data = {
                "roi": self.roi,
                "roi_rotation": self.roi_rotation,
                "axis": self.axis,
                "forward_positive": self.forward_positive,
                "in_count": self.in_count,
                "out_count": self.out_count
            }
            with open(CFG_FILE, "w") as f:
                json.dump(data, f, indent=2)
            self.update_info_signal.emit(f"Saved configuration to {CFG_FILE}")
        except Exception as e:
            self.update_info_signal.emit(f"Error saving config: {str(e)}")
    
    def normalize_roi(self, r):
        if r is None:
            return None
        x1, y1, x2, y2 = r
        if x1 > x2: 
            x1, x2 = x2, x1
        if y1 > y2: 
            y1, y2 = y2, y1
        return [x1, y1, x2, y2]
    
    def get_roi_center(self, r):
        if r is None:
            return None
        x1, y1, x2, y2 = r
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def get_rotated_roi_points(self, r, angle_degrees):
        """Get the four corner points of rotated ROI"""
        if r is None:
            return None
        
        x1, y1, x2, y2 = r
        center_x, center_y = self.get_roi_center(r)
        
        # Calculate the four corners
        corners = [
            (x1, y1),  # Top-left
            (x2, y1),  # Top-right
            (x2, y2),  # Bottom-right
            (x1, y2)   # Bottom-left
        ]
        
        # Convert angle to radians
        angle_rad = math.radians(angle_degrees)
        
        # Rotate each corner around the center
        rotated_corners = []
        for x, y in corners:
            # Translate point to origin
            translated_x = x - center_x
            translated_y = y - center_y
            
            # Apply rotation
            rotated_x = translated_x * math.cos(angle_rad) - translated_y * math.sin(angle_rad)
            rotated_y = translated_x * math.sin(angle_rad) + translated_y * math.cos(angle_rad)
            
            # Translate back
            rotated_x += center_x
            rotated_y += center_y
            
            rotated_corners.append((int(rotated_x), int(rotated_y)))
        
        return rotated_corners
    
    def point_in_rotated_rect(self, pt, r, angle_degrees):
        """Check if point is inside rotated rectangle"""
        if r is None:
            return False
        
        corners = self.get_rotated_roi_points(r, angle_degrees)
        if not corners:
            return False
        
        # Use point-polygon test
        pts = np.array(corners, np.int32)
        result = cv2.pointPolygonTest(pts, pt, False)
        return result >= 0
    
    def get_rotate_handle_position(self, r, angle_degrees):
        """Get position of rotation handle"""
        if r is None:
            return None
        
        center_x, center_y = self.get_roi_center(r)
        
        # Calculate handle position (above center)
        handle_x = center_x
        handle_y = center_y - self.rotate_handle_length
        
        # Rotate handle position
        angle_rad = math.radians(angle_degrees)
        translated_x = handle_x - center_x
        translated_y = handle_y - center_y
        
        rotated_x = translated_x * math.cos(angle_rad) - translated_y * math.sin(angle_rad)
        rotated_y = translated_x * math.sin(angle_rad) + translated_y * math.cos(angle_rad)
        
        rotated_x += center_x
        rotated_y += center_y
        
        return (int(rotated_x), int(rotated_y))
    
    def point_in_rect(self, pt, r):
        if r is None: 
            return False
        x, y = pt
        x1, y1, x2, y2 = r
        return (x1 <= x <= x2) and (y1 <= y <= y2)
    
    def corner_hit(self, pt, r, size=10):
        if r is None: 
            return None
        x, y = pt
        x1, y1, x2, y2 = r
        corners = {
            'tl': (x1, y1),
            'tr': (x2, y1),
            'bl': (x1, y2),
            'br': (x2, y2),
        }
        for name, (cx, cy) in corners.items():
            if abs(x - cx) <= size and abs(y - cy) <= size:
                return name
        return None
    
    def rotate_handle_hit(self, pt, r, angle_degrees, size=10):
        """Check if rotation handle is hit"""
        if r is None:
            return False
        
        handle_pos = self.get_rotate_handle_position(r, angle_degrees)
        if not handle_pos:
            return False
        
        x, y = pt
        hx, hy = handle_pos
        return abs(x - hx) <= size and abs(y - hy) <= size
    
    def move_roi(self, r, dx, dy, w, h):
        x1, y1, x2, y2 = r
        nx1, ny1 = max(0, min(w-1, x1+dx)), max(0, min(h-1, y1+dy))
        nx2, ny2 = max(0, min(w-1, x2+dx)), max(0, min(h-1, y2+dy))
        return [nx1, ny1, nx2, ny2]
    
    def resize_roi(self, r, corner, pt, frame_w, frame_h, min_size=20):
        x1, y1, x2, y2 = r
        x, y = pt
        if corner == 'tl':
            x1, y1 = x, y
        elif corner == 'tr':
            x2, y1 = x, y
        elif corner == 'bl':
            x1, y2 = x, y
        elif corner == 'br':
            x2, y2 = x, y
        
        x1 = max(0, min(frame_w-1, x1))
        x2 = max(0, min(frame_w-1, x2))
        y1 = max(0, min(frame_h-1, y1))
        y2 = max(0, min(frame_h-1, y2))
        r2 = self.normalize_roi([x1, y1, x2, y2])
        
        if r2 and (r2[2]-r2[0] < min_size or r2[3]-r2[1] < min_size):
            return r
        return r2
    
    def rotate_roi(self, r, center, pt, prev_angle):
        """Calculate new rotation angle based on mouse position"""
        if r is None:
            return prev_angle
        
        cx, cy = center
        x, y = pt
        
        # Calculate angle from center to mouse position
        dx = x - cx
        dy = y - cy
        new_angle = math.degrees(math.atan2(dy, dx)) + 90  # +90 to make handle point up at 0 degrees
        
        # Normalize angle to 0-360
        new_angle %= 360
        
        return new_angle
    
    def draw_roi(self, frame, r, color=(0, 255, 0)):
        if r is None: 
            return frame
        
        x1, y1, x2, y2 = r
        center_x, center_y = self.get_roi_center(r)
        
        # Draw rotated rectangle
        if self.roi_rotation != 0:
            corners = self.get_rotated_roi_points(r, self.roi_rotation)
            if corners:
                # Draw filled polygon for better visibility
                pts = np.array(corners, np.int32)
                cv2.polylines(frame, [pts], True, color, 2)
        
        # Draw original rectangle (dashed) for reference when rotated
        if self.roi_rotation != 0:
            # Draw dashed rectangle
            dash_length = 5
            # Top line
            for x in range(x1, x2, dash_length * 2):
                cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), (255, 255, 0), 1)
            # Bottom line
            for x in range(x1, x2, dash_length * 2):
                cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), (255, 255, 0), 1)
            # Left line
            for y in range(y1, y2, dash_length * 2):
                cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), (255, 255, 0), 1)
            # Right line
            for y in range(y1, y2, dash_length * 2):
                cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), (255, 255, 0), 1)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Only draw handles if in edit mode
        if self.edit_mode:
            # Draw resize handles
            for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                cv2.rectangle(frame, (cx-self.handle_size, cy-self.handle_size), 
                             (cx+self.handle_size, cy+self.handle_size), color, -1)
            
            # Draw rotation handle
            handle_pos = self.get_rotate_handle_position(r, self.roi_rotation)
            if handle_pos:
                hx, hy = handle_pos
                # Draw line from center to handle
                cv2.line(frame, (center_x, center_y), (hx, hy), (255, 0, 255), 2)
                # Draw handle circle
                cv2.circle(frame, (hx, hy), self.handle_size, (255, 0, 255), -1)
                # Draw rotation angle text
                cv2.putText(frame, f"{self.roi_rotation:.1f}°", 
                           (hx + 10, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Label IN and OUT sides based on axis and rotation
        if self.axis == 'x':
            # For horizontal counting, IN is left, OUT is right (adjust for rotation)
            in_pos, out_pos = self.get_rotated_text_positions(r, self.roi_rotation, 'x')
            cv2.putText(frame, "IN", in_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "OUT", out_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            # For vertical counting, IN is top, OUT is bottom (adjust for rotation)
            in_pos, out_pos = self.get_rotated_text_positions(r, self.roi_rotation, 'y')
            cv2.putText(frame, "IN", in_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "OUT", out_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame
    
    def get_rotated_text_positions(self, r, angle_degrees, axis):
        """Calculate positions for IN/OUT text based on rotation"""
        center_x, center_y = self.get_roi_center(r)
        x1, y1, x2, y2 = r
        width = x2 - x1
        height = y2 - y1
        
        if axis == 'x':
            # Horizontal counting - IN on left, OUT on right
            in_offset_x = -width // 3
            in_offset_y = 0
            out_offset_x = width // 3
            out_offset_y = 0
        else:
            # Vertical counting - IN on top, OUT on bottom
            in_offset_x = 0
            in_offset_y = -height // 3
            out_offset_x = 0
            out_offset_y = height // 3
        
        # Rotate the offset vectors
        angle_rad = math.radians(angle_degrees)
        
        in_x_rotated = in_offset_x * math.cos(angle_rad) - in_offset_y * math.sin(angle_rad)
        in_y_rotated = in_offset_x * math.sin(angle_rad) + in_offset_y * math.cos(angle_rad)
        
        out_x_rotated = out_offset_x * math.cos(angle_rad) - out_offset_y * math.sin(angle_rad)
        out_y_rotated = out_offset_x * math.sin(angle_rad) + out_offset_y * math.cos(angle_rad)
        
        in_pos = (int(center_x + in_x_rotated - 10), int(center_y + in_y_rotated))
        out_pos = (int(center_x + out_x_rotated - 15), int(center_y + out_y_rotated))
        
        return in_pos, out_pos
    
    def side_of_rect(self, pt, r, axis):
        """Determine which side of the rotated rectangle the point is on"""
        if r is None:
            return 'outside'
        
        # For rotated ROI, we need to transform the point to the ROI's local coordinate system
        center_x, center_y = self.get_roi_center(r)
        x, y = pt
        
        # Translate point to ROI-centered coordinates
        translated_x = x - center_x
        translated_y = y - center_y
        
        # Rotate point back to align with ROI (reverse rotation)
        angle_rad = math.radians(-self.roi_rotation)
        local_x = translated_x * math.cos(angle_rad) - translated_y * math.sin(angle_rad)
        local_y = translated_x * math.sin(angle_rad) + translated_y * math.cos(angle_rad)
        
        x1, y1, x2, y2 = r
        width = x2 - x1
        height = y2 - y1
        
        # Check if point is inside the unrotated ROI bounds
        inside = (-width/2 <= local_x <= width/2) and (-height/2 <= local_y <= height/2)
        
        if inside:
            return 'inside'
        
        if axis == 'x':
            return 'A' if local_x < -width/2 else ('B' if local_x > width/2 else 'inside')
        else:
            return 'A' if local_y < -height/2 else ('B' if local_y > height/2 else 'inside')
    
    def update_counter(self, track_id, pt):
        st = self.obj_state.setdefault(track_id, {'state': 'outside', 'entry_side': None})
        pos = self.side_of_rect(pt, self.roi, self.axis)
        
        if st['state'] == 'outside':
            if pos == 'inside':
                if self.axis == 'x':
                    # Use local x coordinate to determine entry side
                    center_x, center_y = self.get_roi_center(self.roi)
                    translated_x = pt[0] - center_x
                    angle_rad = math.radians(-self.roi_rotation)
                    local_x = translated_x * math.cos(angle_rad)
                    st['entry_side'] = 'A' if local_x < 0 else 'B'
                else:
                    # Use local y coordinate to determine entry side
                    center_x, center_y = self.get_roi_center(self.roi)
                    translated_y = pt[1] - center_y
                    angle_rad = math.radians(-self.roi_rotation)
                    local_y = translated_y * math.cos(angle_rad)
                    st['entry_side'] = 'A' if local_y < 0 else 'B'
                st['state'] = 'inside'
        
        elif st['state'] == 'inside':
            if pos in ('A', 'B'):
                exit_side = pos
                if st['entry_side'] and exit_side != st['entry_side']:
                    # Count IN: A->B or B->A depending on configuration
                    if self.forward_positive:
                        if (st['entry_side'] == 'A' and exit_side == 'B'):
                            self.in_count += 1
                        elif (st['entry_side'] == 'B' and exit_side == 'A'):
                            self.out_count += 1
                    else:
                        if (st['entry_side'] == 'A' and exit_side == 'B'):
                            self.out_count += 1
                        elif (st['entry_side'] == 'B' and exit_side == 'A'):
                            self.in_count += 1
                    
                    self.update_count_signal.emit(self.in_count, self.out_count)
                
                st['state'] = 'outside'
                st['entry_side'] = None
    
    def process_mouse_event(self, event, x, y, flags, frame_w, frame_h):
        if not self.edit_mode:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.roi is not None:
                # Check for rotation handle hit first
                if self.rotate_handle_hit((x, y), self.roi, self.roi_rotation, self.handle_size):
                    self.rotating = True
                    self.mouse_down_pt = (x, y)
                    return
                
                # Check for corner resize handles
                hit = self.corner_hit((x, y), self.roi, self.handle_size)
                if hit:
                    self.resizing = True
                    self.resize_corner = hit
                    return
                
                # Check if clicking inside ROI for dragging
                if self.point_in_rotated_rect((x, y), self.roi, self.roi_rotation):
                    self.dragging = True
                    self.drag_offset = (x - self.roi[0], y - self.roi[1])
                    return
            
            # Start drawing new ROI
            self.mouse_down_pt = (x, y)
            self.roi = [x, y, x, y]
            self.roi_rotation = 0  # Reset rotation for new ROI
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.rotating and self.roi is not None:
                center = self.get_roi_center(self.roi)
                self.roi_rotation = self.rotate_roi(self.roi, center, (x, y), self.roi_rotation)
            elif self.resizing and self.roi is not None:
                self.roi = self.resize_roi(self.roi, self.resize_corner, (x, y), frame_w, frame_h)
            elif self.dragging and self.roi is not None:
                dx = x - (self.roi[0] + self.drag_offset[0])
                dy = y - (self.roi[1] + self.drag_offset[1])
                self.roi = self.move_roi(self.roi, dx, dy, frame_w, frame_h)
            elif self.mouse_down_pt is not None and not self.rotating:
                x0, y0 = self.mouse_down_pt
                self.roi = self.normalize_roi([x0, y0, x, y])
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.resizing = False
            self.rotating = False
            self.resize_corner = None
            self.mouse_down_pt = None
    
    def init_video(self, show_connection_status=False):
        if self.cap_thread is not None:
            self.cap_thread.stop()
        
        self.cap_thread = FrameCaptureThread(self.video_source, show_connection_status=show_connection_status)
        self.cap_thread.new_frame_signal.connect(self.handle_new_frame)
        self.cap_thread.stream_status_signal.connect(self.handle_stream_status)
        self.cap_thread.start()
        
        return True
    
    def handle_new_frame(self, frame):
        pass
    
    def handle_stream_status(self, status, source):
        if status:
            self.connection_status_signal.emit(True, f"Connected to stream: {source}")
        else:
            self.connection_status_signal.emit(False, f"Failed to connect to stream: {source}")
    
    def init_model(self):
        try:
            self.model = YOLO(self.model_path)
            self.update_info_signal.emit(f"Loaded model: {self.model_path}")
            return True
        except Exception as e:
            self.update_info_signal.emit(f"Error loading model: {str(e)}")
            return False
    
    def run(self):
        self.run_flag = True
        
        if not self.init_video(show_connection_status=False):
            self.run_flag = False
            return
        
        if not self.init_model():
            self.run_flag = False
            return
        
        self.load_config()
        
        while self.run_flag:
            if not self.paused and self.cap_thread is not None:
                frame = self.cap_thread.get_frame()
                if frame is None:
                    self.msleep(10)
                    continue
                
                H, W = frame.shape[:2]
                display_frame = frame.copy()
                
                # Process detection and tracking if not in edit mode
                if not self.edit_mode and self.roi is not None:
                    results = self.model.track(
                        source=frame,
                        persist=True,
                        tracker=self.tracker,
                        conf=self.conf_thresh,
                        iou=self.iou_thresh,
                        classes=self.class_filter,  # Only detect people
                        verbose=False
                    )
                    
                    if len(results) > 0:
                        r = results[0]
                        if r.boxes is not None and r.boxes.id is not None:
                            ids = r.boxes.id.cpu().tolist()
                            xyxy = r.boxes.xyxy.cpu().tolist()
                            conf = r.boxes.conf.cpu().tolist() if r.boxes.conf is not None else [None] * len(ids)
                            
                            for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, conf):
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                
                                self.update_counter(int(tid), (cx, cy))
                                
                                # Draw bounding box and ID
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.circle(display_frame, (cx, cy), 3, (0, 0, 255), -1)
                                cv2.putText(display_frame, f"Person {int(tid)}", (x1, y1 - 6),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                
                # Draw ROI if it exists
                if self.roi is not None:
                    display_frame = self.draw_roi(display_frame, self.roi)
                
                # Add info text
                cv2.putText(display_frame, f"IN: {self.in_count}  OUT: {self.out_count}", (40, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                cv2.putText(display_frame,
                           f"Edit:{'ON' if self.edit_mode else 'OFF'}  Axis:{self.axis}  Rot:{self.roi_rotation:.1f}°  Direction:{'A->B=IN' if self.forward_positive else 'B->A=IN'}",
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write the annotated frame to recording
                self.write_frame_to_recording(display_frame)
                
                # Convert to QImage for display
                rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                self.change_pixmap_signal.emit(qt_image)
            
            self.msleep(10)
        
        # Cleanup
        self.stop_recording()
        if self.cap_thread is not None:
            self.cap_thread.stop()
    
    def stop(self):
        self.run_flag = False
        self.stop_recording()
        if self.cap_thread is not None:
            self.cap_thread.stop()
        self.wait()
    
    def toggle_pause(self):
        self.paused = not self.paused
    
    def reset_counter(self):
        self.in_count = 0
        self.out_count = 0
        self.obj_state.clear()
        self.update_count_signal.emit(self.in_count, self.out_count)
    
    def clear_roi(self):
        self.roi = None
        self.roi_rotation = 0
        self.obj_state.clear()
        self.in_count = 0
        self.out_count = 0
        self.update_count_signal.emit(self.in_count, self.out_count)

# =========================
# Main GUI Window
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("People Counter - Person In/Out Tracking System")
        
        # MQTT and server state
        self.last_sent_count = (-1, -1)
        self.video_path = ""
        
        # Get screen dimensions
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        # Set window size based on screen dimensions
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        self.setGeometry(100, 100, window_width, window_height)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create a splitter for resizable panels
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout = QHBoxLayout(central_widget)
        main_layout.addWidget(main_splitter)
        
        # Left panel for controls (scrollable)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMaximumWidth(int(window_width * 0.3))
        left_scroll.setMinimumWidth(250)
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        
        # Camera selection section
        camera_group = QGroupBox("📷 Camera Selection")
        camera_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        camera_layout = QVBoxLayout(camera_group)
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(list(CAMERA_OPTIONS.keys()))
        self.camera_combo.currentTextChanged.connect(self.on_camera_changed)
        
        self.rtsp_edit = QLineEdit(DEFAULT_VIDEO_SOURCE)
        self.connect_btn = QPushButton("🔗 Connect RTSP")
        self.connect_btn.clicked.connect(self.connect_rtsp)
        
        camera_layout.addWidget(QLabel("Select Camera:"))
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(QLabel("RTSP URL:"))
        camera_layout.addWidget(self.rtsp_edit)
        camera_layout.addWidget(self.connect_btn)
        
        # ROI settings section
        roi_group = QGroupBox("🎯 ROI Settings")
        roi_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        roi_layout = QVBoxLayout(roi_group)
        
        self.draw_roi_btn = QPushButton("📐 Enable ROI Drawing")
        self.draw_roi_btn.clicked.connect(self.toggle_edit_mode)
        
        self.clear_roi_btn = QPushButton("🗑️ Clear ROI")
        self.clear_roi_btn.clicked.connect(self.clear_roi)
        
        # Rotation controls
        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(QLabel("Rotation:"))
        self.rotation_slider_label = QLabel("0°")
        self.rotation_slider = QDoubleSpinBox()
        self.rotation_slider.setRange(0, 359.9)
        self.rotation_slider.setSingleStep(1.0)
        self.rotation_slider.setValue(0)
        self.rotation_slider.valueChanged.connect(self.on_rotation_changed)
        rotation_layout.addWidget(self.rotation_slider)
        rotation_layout.addWidget(self.rotation_slider_label)
        
        # Axis selection
        axis_layout = QHBoxLayout()
        axis_layout.addWidget(QLabel("Counting Axis:"))
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["Vertical (Up/Down)", "Horizontal (Left/Right)"])
        self.axis_combo.currentIndexChanged.connect(self.on_axis_changed)
        axis_layout.addWidget(self.axis_combo)
        
        # Direction selection
        direction_layout = QHBoxLayout()
        direction_layout.addWidget(QLabel("IN Direction:"))
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["A→B = IN, B→A = OUT", "B→A = IN, A→B = OUT"])
        self.direction_combo.currentIndexChanged.connect(self.on_direction_changed)
        direction_layout.addWidget(self.direction_combo)
        
        roi_layout.addWidget(self.draw_roi_btn)
        roi_layout.addWidget(self.clear_roi_btn)
        roi_layout.addLayout(rotation_layout)
        roi_layout.addLayout(axis_layout)
        roi_layout.addLayout(direction_layout)
        
        # Recording settings section
        recording_group = QGroupBox("🎥 Recording Settings")
        recording_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        recording_layout = QVBoxLayout(recording_group)
        
        self.recording_status_label = QLabel("Status: Ready")
        self.recordings_path_label = QLabel(f"Saved to: {RECORDINGS_DIR}/")
        self.recordings_path_label.setWordWrap(True)
        self.recordings_path_label.setStyleSheet("font-size: 8pt; color: #AAAAAA;")
        
        recording_layout.addWidget(QLabel("Recording Status:"))
        recording_layout.addWidget(self.recording_status_label)
        recording_layout.addWidget(self.recordings_path_label)
        
        # MQTT Settings section
        mqtt_group = QGroupBox("📡 MQTT Settings")
        mqtt_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        mqtt_layout = QVBoxLayout(mqtt_group)
        
        self.mqtt_status_label = QLabel("Status: Disconnected")
        self.mqtt_broker_edit = QLineEdit(MQTT_BROKER)
        self.mqtt_topic_edit = QLineEdit(MQTT_TOPIC)
        
        mqtt_layout.addWidget(QLabel("MQTT Broker:"))
        mqtt_layout.addWidget(self.mqtt_broker_edit)
        mqtt_layout.addWidget(QLabel("MQTT Topic:"))
        mqtt_layout.addWidget(self.mqtt_topic_edit)
        mqtt_layout.addWidget(self.mqtt_status_label)
        
        # Control buttons section
        control_group = QGroupBox("🎮 Controls")
        control_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        control_layout = QVBoxLayout(control_group)
        
        self.start_btn = QPushButton("▶️ Start Counting")
        self.start_btn.clicked.connect(self.start_counting)
        
        self.stop_btn = QPushButton("⏹️ Stop Counting")
        self.stop_btn.clicked.connect(self.stop_counting)
        self.stop_btn.setEnabled(False)
        
        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        
        self.reset_btn = QPushButton("🔄 Reset Counter")
        self.reset_btn.clicked.connect(self.reset_counter)
        
        self.save_cfg_btn = QPushButton("💾 Save Configuration")
        self.save_cfg_btn.clicked.connect(self.save_config)
        
        self.load_cfg_btn = QPushButton("📂 Load Configuration")
        self.load_cfg_btn.clicked.connect(self.load_config)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.reset_btn)
        control_layout.addWidget(self.save_cfg_btn)
        control_layout.addWidget(self.load_cfg_btn)
        
        # Status section
        status_group = QGroupBox("📊 Status")
        status_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        status_layout = QVBoxLayout(status_group)
        
        self.in_count_label = QLabel("IN: 0")
        self.in_count_label.setStyleSheet("font-weight: bold; font-size: 40px; color: #4CAF50;")
        
        self.out_count_label = QLabel("OUT: 0")
        self.out_count_label.setStyleSheet("font-weight: bold; font-size: 40px; color: #F44336;")
        
        self.total_label = QLabel("TOTAL: 0")
        self.total_label.setStyleSheet("font-weight: bold; font-size: 30px; color: #2196F3;")
        
        self.info_label = QLabel("Ready to start")
        self.info_label.setWordWrap(True)
        
        status_layout.addWidget(self.in_count_label)
        status_layout.addWidget(self.out_count_label)
        status_layout.addWidget(self.total_label)
        status_layout.addWidget(self.info_label)
        
        # Add all groups to left layout
        left_layout.addWidget(camera_group)
        left_layout.addWidget(status_group)
        left_layout.addWidget(roi_group)
        left_layout.addWidget(control_group)
        left_layout.addWidget(recording_group)
        left_layout.addWidget(mqtt_group)
        left_layout.addStretch()
        
        # Set left panel to scroll area
        left_scroll.setWidget(left_panel)
        
        # Right panel for video display
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setText("Video will appear here")
        self.video_label.setStyleSheet("""
            border: 2px solid #4A4A4A; 
            background-color: #1E1E1E; 
            border-radius: 5px;
            color: #FFFFFF;
            font-weight: bold;
        """)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        video_layout.addWidget(self.video_label)
        
        # Add panels to splitter
        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(video_container)
        
        # Set splitter sizes (30% for controls, 70% for video)
        splitter_sizes = [int(window_width * 0.3), int(window_width * 0.7)]
        main_splitter.setSizes(splitter_sizes)
        
        # Initialize video thread
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_count_signal.connect(self.update_count)
        self.thread.update_info_signal.connect(self.update_info)
        self.thread.connection_status_signal.connect(self.handle_connection_status)
        self.thread.recording_status_signal.connect(self.handle_recording_status)
        
        # Initialize MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
        # Start MQTT
        self.start_mqtt()
        
        # Set initial UI state
        self.on_camera_changed("Camera 1")
        
        # Apply styles
        self.apply_styles()
    
    def apply_styles(self):
        # Set font sizes based on screen resolution
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_height = screen_geometry.height()
        
        base_font_size = 9 if screen_height > 1080 else 8
        
        # Main window style (same as before)
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: #2C2C2C;
            }}
            
            QWidget {{
                background-color: #2C2C2C;
                color: #FFFFFF;
            }}
            
            QGroupBox {{
                font-weight: bold;
                font-size: {base_font_size + 1}pt;
                color: #FFFFFF;
                border: 2px solid #4A4A4A;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: #3C3C3C;
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #FFD700;
            }}
            
            QPushButton {{
                font-size: {base_font_size}pt;
                padding: 8px;
                min-height: 30px;
                border: 1px solid #555555;
                border-radius: 5px;
                background-color: #4A4A4A;
                color: #FFFFFF;
            }}
            
            QPushButton:hover {{
                background-color: #5A5A5A;
                border: 1px solid #666666;
            }}
            
            QPushButton:pressed {{
                background-color: #3A3A3A;
            }}
            
            QPushButton:disabled {{
                background-color: #333333;
                color: #888888;
            }}
            
            QLabel {{
                font-size: {base_font_size}pt;
                color: #DDDDDD;
            }}
            
            QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
                font-size: {base_font_size}pt;
                padding: 5px;
                border: 1px solid #555555;
                border-radius: 4px;
                background-color: #3C3C3C;
                color: #FFFFFF;
                selection-background-color: #555555;
            }}
            
            QComboBox:editable, QLineEdit {{
                background-color: #3C3C3C;
            }}
            
            QComboBox QAbstractItemView {{
                background-color: #3C3C3C;
                color: #FFFFFF;
                selection-background-color: #555555;
            }}
            
            QScrollArea {{
                border: none;
                background-color: #2C2C2C;
            }}
            
            QScrollBar:vertical {{
                border: none;
                background-color: #3C3C3C;
                width: 10px;
                margin: 0px;
            }}
            
            QScrollBar::handle:vertical {{
                background-color: #5A5A5A;
                min-height: 20px;
                border-radius: 5px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background-color: #6A6A6A;
            }}
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        
        # Special styling for specific buttons (same as before)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border: 1px solid #3E8E41;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            QPushButton:pressed {
                background-color: #3E8E41;
            }
        """)
        
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                font-weight: bold;
                border: 1px solid #D32F2F;
            }
            QPushButton:hover {
                background-color: #E53935;
            }
            QPushButton:pressed {
                background-color: #D32F2F;
            }
        """)
        
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                border: 1px solid #F57C00;
            }
            QPushButton:hover {
                background-color: #FB8C00;
            }
            QPushButton:pressed {
                background-color: #F57C00;
            }
        """)
        
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border: 1px solid #0B7DDA;
            }
            QPushButton:hover {
                background-color: #1E88E5;
            }
            QPushButton:pressed {
                background-color: #0B7DDA;
            }
        """)
    
    # MQTT Methods (same as before)
    def start_mqtt(self):
        try:
            mqtt_thread = threading.Thread(target=self.run_mqtt, daemon=True)
            mqtt_thread.start()
            self.update_info("MQTT client starting...")
        except Exception as e:
            self.update_info(f"MQTT thread start failed: {e}")

    def run_mqtt(self):
        try:
            broker = self.mqtt_broker_edit.text().strip() or MQTT_BROKER
            self.mqtt_client.connect(broker, MQTT_PORT, 60)
            self.mqtt_client.loop_forever()
        except Exception as e:
            self.update_info(f"MQTT connection failed: {e}")

    def on_mqtt_connect(self, client, userdata, flags, rc):
        topic = self.mqtt_topic_edit.text().strip() or MQTT_TOPIC
        if rc == 0:
            client.subscribe(topic)
            self.mqtt_status_label.setText("Status: Connected")
            self.update_info(f"MQTT connected to topic: {topic}")
        else:
            self.mqtt_status_label.setText(f"Status: Connection failed (code {rc})")
            self.update_info(f"MQTT connection failed with code {rc}")

    def on_mqtt_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode()
            self.update_info(f"MQTT message received: {payload}")
            data = json.loads(payload)
            action = data.get("action")
            
            if action == "start":
                self.update_info("MQTT Action: Start")
                QTimer.singleShot(0, self.start_counting)
                
            elif action == "stop":
                self.update_info("MQTT Action: Stop")
                QTimer.singleShot(0, self.stop_counting)
                
            elif action == "reset":
                self.update_info("MQTT Action: Reset")
                QTimer.singleShot(0, self.reset_counter)

        except Exception as e:
            self.update_info(f"MQTT message error: {e}")

    def on_camera_changed(self, camera_name):
        if camera_name == "Custom RTSP":
            self.rtsp_edit.setEnabled(True)
            self.rtsp_edit.setText("")
        else:
            self.rtsp_edit.setEnabled(False)
            self.rtsp_edit.setText(CAMERA_OPTIONS[camera_name])
        
        self.connect_btn.setEnabled(True)
    
    def connect_rtsp(self):
        rtsp_url = self.rtsp_edit.text().strip()
        if rtsp_url:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.release()
                self.show_connection_popup(True, f"Successfully connected to RTSP stream:\n{rtsp_url}")
                self.update_info(f"Connected to RTSP stream: {rtsp_url}")
                self.video_path = rtsp_url
                
                if hasattr(self, 'thread'):
                    self.thread.video_source = rtsp_url
                    if self.thread.isRunning():
                        self.thread.stop()
                        self.thread.init_video(show_connection_status=True)
            else:
                self.show_connection_popup(False, f"Failed to connect to RTSP stream:\n{rtsp_url}")
                self.update_info(f"Failed to connect to RTSP stream: {rtsp_url}")
        else:
            self.show_connection_popup(False, "Please enter a valid RTSP URL")
            self.update_info("Please enter a valid RTSP URL")
    
    def show_connection_popup(self, success, message):
        msg = QMessageBox(self)
        msg.setWindowTitle("Connection Status")
        msg.setText(message)
        
        if success:
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        else:
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        
        msg.exec()
    
    @pyqtSlot(bool, str)
    def handle_connection_status(self, status, message):
        self.show_connection_popup(status, message)
        self.update_info(message)
    
    @pyqtSlot(bool, str)
    def handle_recording_status(self, is_recording, message):
        if is_recording:
            self.recording_status_label.setText("Status: Recording...")
            self.recording_status_label.setStyleSheet("color: #FF4444; font-weight: bold;")
        else:
            self.recording_status_label.setText("Status: Ready")
            self.recording_status_label.setStyleSheet("color: #DDDDDD;")
        self.update_info(message)
    
    def on_rotation_changed(self, value):
        self.thread.roi_rotation = value
        self.rotation_slider_label.setText(f"{value:.1f}°")
        self.update_info(f"ROI rotation set to {value:.1f} degrees")
    
    def on_axis_changed(self, index):
        self.thread.axis = 'x' if index == 1 else 'y'
        self.update_info(f"Counting axis changed to {'horizontal' if index == 1 else 'vertical'}")
    
    def on_direction_changed(self, index):
        self.thread.forward_positive = (index == 0)
        direction = "A→B = IN, B→A = OUT" if index == 0 else "B→A = IN, A→B = OUT"
        self.update_info(f"Counting direction changed to: {direction}")
    
    def toggle_edit_mode(self):
        self.thread.edit_mode = not self.thread.edit_mode
        if self.thread.edit_mode:
            self.draw_roi_btn.setText("🛑 Disable ROI Drawing")
            self.draw_roi_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF6B6B;
                    color: white;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #FF5252;
                }
            """)
            self.update_info("ROI drawing mode enabled. Click and drag on video to draw ROI. Drag rotation handle to rotate.")
        else:
            self.draw_roi_btn.setText("📐 Enable ROI Drawing")
            self.draw_roi_btn.setStyleSheet("")
            self.update_info("ROI drawing mode disabled.")
    
    def clear_roi(self):
        self.thread.clear_roi()
        self.rotation_slider.setValue(0)
        self.update_info("ROI cleared")
    
    def start_counting(self):
        self.thread.video_source = self.rtsp_edit.text()
        self.thread.start_recording()
        self.thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.video_label.mousePressEvent = self.mouse_press_event
        self.video_label.mouseMoveEvent = self.mouse_move_event
        self.video_label.mouseReleaseEvent = self.mouse_release_event
        
        self.update_info("People counting and recording started")
    
    def stop_counting(self):
        self.thread.stop_recording()
        self.thread.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.video_label.mousePressEvent = None
        self.video_label.mouseMoveEvent = None
        self.video_label.mouseReleaseEvent = None
        
        self.update_info("People counting and recording stopped")
    
    def toggle_pause(self):
        self.thread.toggle_pause()
        if self.thread.paused:
            self.pause_btn.setText("▶️ Resume")
            self.update_info("Counting paused")
        else:
            self.pause_btn.setText("⏸️ Pause")
            self.update_info("Counting resumed")
    
    def reset_counter(self):
        self.thread.reset_counter()
        self.update_info("Counter reset")
    
    def save_config(self):
        self.thread.save_config()
    
    def load_config(self):
        self.thread.load_config()
        # Update UI to match loaded config
        self.axis_combo.setCurrentIndex(0 if self.thread.axis == 'y' else 1)
        self.direction_combo.setCurrentIndex(0 if self.thread.forward_positive else 1)
        self.rotation_slider.setValue(self.thread.roi_rotation)
        self.update_count(self.thread.in_count, self.thread.out_count)
    
    @pyqtSlot(QImage)
    def update_image(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.video_label.width(), 
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
    
    @pyqtSlot(int, int)
    def update_count(self, in_count, out_count):
        self.in_count_label.setText(f"IN: {in_count}")
        self.out_count_label.setText(f"OUT: {out_count}")
        total = in_count - out_count
        self.total_label.setText(f"TOTAL: {total}")
    
    @pyqtSlot(str)
    def update_info(self, message):
        self.info_label.setText(message)
        print(f"INFO: {message}")
    
    def mouse_press_event(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.thread.edit_mode:
            x = event.pos().x()
            y = event.pos().y()
            
            pixmap = self.video_label.pixmap()
            if pixmap:
                label_width = self.video_label.width()
                label_height = self.video_label.height()
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                
                x_offset = (label_width - pixmap_width) // 2 if label_width > pixmap_width else 0
                y_offset = (label_height - pixmap_height) // 2 if label_height > pixmap_height else 0
                
                x = max(0, min(pixmap_width - 1, x - x_offset))
                y = max(0, min(pixmap_height - 1, y - y_offset))
                
                x = int(x * 640 / pixmap_width)
                y = int(y * 480 / pixmap_height)
                
                self.thread.process_mouse_event(
                    cv2.EVENT_LBUTTONDOWN, x, y, 0, 640, 480
                )
    
    def mouse_move_event(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.thread.edit_mode:
            x = event.pos().x()
            y = event.pos().y()
            
            pixmap = self.video_label.pixmap()
            if pixmap:
                label_width = self.video_label.width()
                label_height = self.video_label.height()
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                
                x_offset = (label_width - pixmap_width) // 2 if label_width > pixmap_width else 0
                y_offset = (label_height - pixmap_height) // 2 if label_height > pixmap_height else 0
                
                x = max(0, min(pixmap_width - 1, x - x_offset))
                y = max(0, min(pixmap_height - 1, y - y_offset))
                
                x = int(x * 640 / pixmap_width)
                y = int(y * 480 / pixmap_height)
                
                self.thread.process_mouse_event(
                    cv2.EVENT_MOUSEMOVE, x, y, 0, 640, 480
                )
    
    def mouse_release_event(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.thread.edit_mode:
            x = event.pos().x()
            y = event.pos().y()
            
            pixmap = self.video_label.pixmap()
            if pixmap:
                label_width = self.video_label.width()
                label_height = self.video_label.height()
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                
                x_offset = (label_width - pixmap_width) // 2 if label_width > pixmap_width else 0
                y_offset = (label_height - pixmap_height) // 2 if label_height > pixmap_height else 0
                
                x = max(0, min(pixmap_width - 1, x - x_offset))
                y = max(0, min(pixmap_height - 1, y - y_offset))
                
                x = int(x * 640 / pixmap_width)
                y = int(y * 480 / pixmap_height)
                
                self.thread.process_mouse_event(
                    cv2.EVENT_LBUTTONUP, x, y, 0, 640, 480
                )
    
    def resizeEvent(self, event):
        if hasattr(self, 'thread') and self.thread.isRunning():
            pixmap = self.video_label.pixmap()
            if pixmap and not pixmap.isNull():
                self.video_label.setPixmap(pixmap.scaled(
                    self.video_label.width(),
                    self.video_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
        super().resizeEvent(event)
    
    def closeEvent(self, event):
        try:
            if hasattr(self, 'thread'):
                self.thread.stop_recording()
                self.thread.stop()
            
            if hasattr(self, 'mqtt_client'):
                self.mqtt_client.disconnect()
            
            if os.path.exists(RECORDINGS_DIR) and os.listdir(RECORDINGS_DIR):
                files = os.listdir(RECORDINGS_DIR)
                msg = QMessageBox(self)
                msg.setWindowTitle("Recordings Saved")
                msg.setText(f"All recordings have been saved to:\n{os.path.abspath(RECORDINGS_DIR)}\n\nFiles: {', '.join(files[-3:])}")
                msg.setIcon(QMessageBox.Icon.Information)
                msg.exec()
            
        except Exception as e:
            print(f"Error during application close: {e}")
        
        event.accept()

# =========================
# Application Entry Point
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())