import sys
import cv2
import json
import os
import queue
import time
import threading
import requests
import gc
import psutil
import traceback
import random
from collections import defaultdict, deque
from ultralytics import YOLO
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QLineEdit, QGroupBox, 
                             QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox,
                             QSizePolicy, QScrollArea, QSplitter, QTabWidget)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
import paho.mqtt.client as mqtt
from datetime import datetime
from threading import Lock

# =========================
# Configuration
# =========================
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

DEFAULT_MODEL_PATH = resource_path(r"C:\Users\ayuba\Downloads\best (21).pt")
DEFAULT_VIDEO_SOURCE = r"rtsp://admin:pass%40123@192.168.1.240:554/cam/realmonitor?channel=1&subtype=0"
CFG_FILE = "roi_config_wagon1.json"
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "cement/wagon/control/1"
MQTT_TOPIC_CURRENT_COUNT = 'cement/wagon/cantrol/current_count'
MQTT_TOPIC_TARGET_REACHED = 'cement/wagon/control/target_reached'
SERVER_URL = "https://shipeasy.tech/cement/public/api/get_load"
RECORDINGS_DIR = "recordings_wagon"

# Frame dimensions
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Predefined camera options
CAMERA_OPTIONS = {
    "Camera 1": "rtsp://admin:Admin%4012345@192.168.53.127:554/cam/realmonitor?channel=1&subtype=0",
    "Camera 2": "rtsp://admin:Admin%4012345@192.168.53.137:554/cam/realmonitor?channel=1&subtype=0",
    "Camera 3": "rtsp://admin:Admin%4012345@192.168.53.126:554/cam/realmonitor?channel=1&subtype=0",
    "Camera 4": "rtsp://admin:Admin%4012345@192.168.53.128:554/cam/realmonitor?channel=1&subtype=0",
    "Camera 5": "rtsp://localhost:8554/mystream",
    "Custom RTSP": ""
}

# =========================
# Enhanced Frame Capture Thread
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
        self.FRAME_WIDTH = FRAME_WIDTH
        self.FRAME_HEIGHT = FRAME_HEIGHT
        self.error_count = 0
        self.max_errors = 5
    
    def run(self):
        self.running = True
        cap = None
        
        try:
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
            while self.running and self.error_count < self.max_errors:
                ret, frame = cap.read()
                if not ret:
                    self.error_count += 1
                    print(f"Frame grab failed for {self.video_source} (error {self.error_count}/{self.max_errors})")
                    time.sleep(1)
                    continue
                
                # Reset error count on successful frame
                self.error_count = 0
                
                # Resize frame to fixed dimensions
                frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
                self.new_frame_signal.emit(frame)
                
                # Add to queue for processing
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, timeout=1)
                except queue.Full:
                    pass
                except Exception as e:
                    print(f"Error adding frame to queue: {e}")

        except Exception as e:
            print(f"Error in frame capture thread: {e}")
        finally:
            if cap is not None:
                cap.release()
            if self.show_connection_status:
                self.stream_status_signal.emit(False, self.video_source)
            self.is_stream_active = False
    
    def stop(self):
        self.running = False
        self.wait(2000)  # Wait up to 2 seconds
    
    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

# =========================
# Enhanced Worker Thread for Video Processing
# =========================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)
    update_count_signal = pyqtSignal(int)
    update_info_signal = pyqtSignal(str)
    connection_status_signal = pyqtSignal(bool, str)
    recording_status_signal = pyqtSignal(bool, str)
    secondary_frame_signal = pyqtSignal(QImage)
    target_reached_signal = pyqtSignal(int, int)
    counting_status_signal = pyqtSignal(bool)
    memory_warning_signal = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.run_flag = True
        self.paused = False
        self.edit_mode = False
        
        # Enhanced state management
        self.last_count_publish_time = 0
        self.count_publish_interval = 3
        
        # Configuration parameters
        self.video_source = DEFAULT_VIDEO_SOURCE
        self.secondary_video_source = ""
        self.model_path = DEFAULT_MODEL_PATH
        self.class_filter = None
        self.conf_thresh = 0.5
        self.iou_thresh = 0.45
        self.tracker = "bytetrack.yaml"
        self.axis = 'y'
        self.forward_positive = True
        
        # Enhanced ROI and tracking state with memory management
        self.roi = None
        self.obj_state = {}
        self.max_obj_state_size = 2000
        self.count_value = 0
        self.target_count = 0
        self.target_reached = False
        
        # Thread safety
        self.state_lock = Lock()
        self.frame_lock = Lock()
        
        # Mouse interaction state
        self.dragging = False
        self.resizing = False
        self.drag_offset = (0, 0)
        self.resize_corner = None
        self.mouse_down_pt = None
        self.handle_size = 10
        
        # Video capture and model
        self.cap_thread = None
        self.secondary_cap_thread = None
        self.model = None
        
        # Recording state with error handling
        self.recording = False
        self.primary_video_writer = None
        self.secondary_video_writer = None
        self.primary_recording_filename = None
        self.secondary_recording_filename = None
        
        # Active camera for detection
        self.active_camera = "primary"
        
        # Frame dimensions
        self.FRAME_WIDTH = FRAME_WIDTH
        self.FRAME_HEIGHT = FRAME_HEIGHT
        
        # Memory management
        self.last_memory_check = 0
        self.memory_check_interval = 30
        self.high_memory_count = 0
        
        # Create recordings directory if it doesn't exist
        if not os.path.exists(RECORDINGS_DIR):
            os.makedirs(RECORDINGS_DIR)

    def safe_cleanup(self):
        """Comprehensive cleanup method"""
        try:
            self.run_flag = False
            self.paused = True
            
            # Stop recording first
            self.stop_recording()
            
            # Stop capture threads
            if self.cap_thread is not None:
                self.cap_thread.stop()
                
            if self.secondary_cap_thread is not None:
                self.secondary_cap_thread.stop()
                
            # Clear large data structures
            with self.state_lock:
                self.obj_state.clear()
                
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Error during safe_cleanup: {e}")

    def check_memory_usage(self):
        """Monitor and manage memory usage"""
        try:
            current_time = time.time()
            if current_time - self.last_memory_check < self.memory_check_interval:
                return
                
            self.last_memory_check = current_time
            
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            # Emit warning if memory is high
            if memory_mb > 800:
                self.high_memory_count += 1
                self.memory_warning_signal.emit(memory_mb)
                self.update_info_signal.emit(f"⚠️ High memory usage: {memory_mb:.1f}MB")
                
                # Aggressive cleanup at very high memory
                if memory_mb > 1200 or self.high_memory_count > 3:
                    self.perform_memory_cleanup()
                    self.high_memory_count = 0
                    
        except Exception as e:
            print(f"Memory check error: {e}")

    def perform_memory_cleanup(self):
        """Perform aggressive memory cleanup"""
        try:
            self.update_info_signal.emit("🔄 Performing memory cleanup...")
            
            # Clean object state
            with self.state_lock:
                if len(self.obj_state) > 1000:
                    keys_to_keep = list(self.obj_state.keys())[-500:]
                    self.obj_state = {k: self.obj_state[k] for k in keys_to_keep}
            
            # Force garbage collection
            gc.collect()
            
            self.update_info_signal.emit("✅ Memory cleanup completed")
            
        except Exception as e:
            self.update_info_signal.emit(f"❌ Memory cleanup failed: {e}")

    def set_target_count(self, target):
        """Set the target count and reset target reached flag"""
        self.target_count = target
        self.target_reached = False
        self.update_info_signal.emit(f"Target count set to: {target}")

    def set_active_camera(self, camera):
        """Set which camera to use for detection"""
        self.active_camera = camera
        self.update_info_signal.emit(f"Active detection camera: {camera}")

    def start_recording(self):
        """Start video recording for both cameras"""
        if not self.recording:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Primary camera recording
                self.primary_recording_filename = os.path.join(RECORDINGS_DIR, f"primary_recording_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.primary_video_writer = cv2.VideoWriter(
                    self.primary_recording_filename, 
                    fourcc, 
                    15.0,
                    (self.FRAME_WIDTH, self.FRAME_HEIGHT)
                )
                
                # Secondary camera recording
                self.secondary_recording_filename = os.path.join(RECORDINGS_DIR, f"secondary_recording_{timestamp}.mp4")
                self.secondary_video_writer = cv2.VideoWriter(
                    self.secondary_recording_filename, 
                    fourcc, 
                    15.0,
                    (self.FRAME_WIDTH, self.FRAME_HEIGHT)
                )
                
                if self.primary_video_writer.isOpened() and self.secondary_video_writer.isOpened():
                    self.recording = True
                    self.recording_status_signal.emit(True, f"Recording started for both cameras")
                else:
                    self.recording_status_signal.emit(False, "Failed to start recording for one or both cameras")
                    
            except Exception as e:
                self.recording_status_signal.emit(False, f"Error starting recording: {e}")
  
    def stop_recording(self):
        """Stop video recording for both cameras"""
        if self.recording:
            try:
                if self.primary_video_writer is not None:
                    self.primary_video_writer.release()
                    self.primary_video_writer = None
                
                if self.secondary_video_writer is not None:
                    self.secondary_video_writer.release()
                    self.secondary_video_writer = None
                
                self.recording = False
                self.recording_status_signal.emit(False, f"Recordings saved: Primary - {self.primary_recording_filename}, Secondary - {self.secondary_recording_filename}")
            except Exception as e:
                self.recording_status_signal.emit(False, f"Error stopping recording: {e}")

    def write_frame_to_recording(self, primary_frame, secondary_frame):
        """Write frames to both camera recordings if recording is active"""
        if self.recording:
            try:
                if self.primary_video_writer is not None:
                    self.primary_video_writer.write(primary_frame)
                
                if self.secondary_video_writer is not None and secondary_frame is not None:
                    self.secondary_video_writer.write(secondary_frame)
            except Exception as e:
                print(f"Error writing frame to recording: {e}")

    def load_config(self):
        if os.path.exists(CFG_FILE):
            try:
                with open(CFG_FILE, "r") as f:
                    data = json.load(f)
                self.roi = self.normalize_roi(data.get("roi"))
                self.axis = data.get("axis", self.axis)
                self.forward_positive = data.get("forward_positive", self.forward_positive)
                self.target_count = data.get("target_count", 0)
                self.update_info_signal.emit(f"Loaded configuration from {CFG_FILE}")
            except Exception as e:
                self.update_info_signal.emit(f"Error loading config: {str(e)}")
    
    def save_config(self):
        try:
            data = {
                "roi": self.roi,
                "axis": self.axis,
                "forward_positive": self.forward_positive,
                "target_count": self.target_count
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
    
    def draw_roi(self, frame, r, color=(0, 255, 0)):
        if r is None: 
            return frame
        
        x1, y1, x2, y2 = r
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        if self.edit_mode:
            for (cx, cy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                cv2.rectangle(frame, (cx-self.handle_size, cy-self.handle_size), 
                             (cx+self.handle_size, cy+self.handle_size), color, -1)
        
        if self.axis == 'x':
            cv2.putText(frame, "IN", (x1+5, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "OUT", (x2-20, (y1+y2)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "IN", ((x1+x2)//2-10, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "OUT", ((x1+x2)//2-10, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame
    
    def side_of_rect(self, pt, r, axis):
        if r is None:
            return 'outside'
        x, y = pt
        x1, y1, x2, y2 = r
        inside = (x1 <= x <= x2) and (y1 <= y <= y2)
        if inside:
            return 'inside'
        if axis == 'x':
            return 'A' if x < x1 else ('B' if x > x2 else 'inside')
        else:
            return 'A' if y < y1 else ('B' if y > y2 else 'inside')
    
    def update_counter(self, track_id, pt):
        """Enhanced counter with memory management"""
        try:
            with self.state_lock:
                # Clean old entries if needed
                if len(self.obj_state) > self.max_obj_state_size:
                    keys_to_remove = list(self.obj_state.keys())[:self.max_obj_state_size // 5]
                    for key in keys_to_remove:
                        del self.obj_state[key]
                
                st = self.obj_state.setdefault(track_id, {'state': 'outside', 'entry_side': None})
                
                pos = self.side_of_rect(pt, self.roi, self.axis)
                
                if st['state'] == 'outside':
                    if pos == 'inside':
                        if self.axis == 'x':
                            x = pt[0]
                            x1, y1, x2, y2 = self.roi
                            st['entry_side'] = 'A' if abs(x - x1) < abs(x - x2) else 'B'
                        else:
                            y = pt[1]
                            x1, y1, x2, y2 = self.roi
                            st['entry_side'] = 'A' if abs(y - y1) < abs(y - y2) else 'B'
                        st['state'] = 'inside'
                
                elif st['state'] == 'inside':
                    if pos in ('A', 'B'):
                        exit_side = pos
                        if st['entry_side'] and exit_side != st['entry_side']:
                            if self.forward_positive:
                                delta = +1 if (st['entry_side'] == 'A' and exit_side == 'B') else -1
                            else:
                                delta = -1 if (st['entry_side'] == 'A' and exit_side == 'B') else +1
                            self.count_value += delta
                            self.update_count_signal.emit(self.count_value)
                            
                            # Safe target check
                            if (self.target_count > 0 and 
                                not self.target_reached and 
                                self.count_value >= self.target_count):
                                self.target_reached = True
                                self.target_reached_signal.emit(self.count_value, self.target_count)
                        
                        st['state'] = 'outside'
                        st['entry_side'] = None
                        
        except Exception as e:
            print(f"Error in update_counter: {e}")
    
    def process_mouse_event(self, event, x, y, flags, frame_w, frame_h):
        if not self.edit_mode:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.roi is not None:
                hit = self.corner_hit((x, y), self.roi, self.handle_size)
                if hit:
                    self.resizing = True
                    self.resize_corner = hit
                    return
                if self.point_in_rect((x, y), self.roi):
                    self.dragging = True
                    self.drag_offset = (x - self.roi[0], y - self.roi[1])
                    return
            
            self.mouse_down_pt = (x, y)
            self.roi = [x, y, x, y]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.resizing and self.roi is not None:
                self.roi = self.resize_roi(self.roi, self.resize_corner, (x, y), frame_w, frame_h)
            elif self.dragging and self.roi is not None:
                dx = x - (self.roi[0] + self.drag_offset[0])
                dy = y - (self.roi[1] + self.drag_offset[1])
                self.roi = self.move_roi(self.roi, dx, dy, frame_w, frame_h)
            elif self.mouse_down_pt is not None:
                x0, y0 = self.mouse_down_pt
                self.roi = self.normalize_roi([x0, y0, x, y])
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.resizing = False
            self.resize_corner = None
            self.mouse_down_pt = None
    
    def init_video(self, show_connection_status=False):
        # Initialize primary camera
        if self.cap_thread is not None:
            self.cap_thread.stop()
        
        self.cap_thread = FrameCaptureThread(self.video_source, show_connection_status=show_connection_status)
        self.cap_thread.new_frame_signal.connect(self.handle_primary_frame)
        self.cap_thread.stream_status_signal.connect(self.handle_stream_status)
        self.cap_thread.start()
        
        # Initialize secondary camera if URL is provided
        if self.secondary_video_source:
            if self.secondary_cap_thread is not None:
                self.secondary_cap_thread.stop()
            
            self.secondary_cap_thread = FrameCaptureThread(self.secondary_video_source, show_connection_status=False)
            self.secondary_cap_thread.new_frame_signal.connect(self.handle_secondary_frame)
            self.secondary_cap_thread.start()
        
        return True
    
    def handle_primary_frame(self, frame):
        pass
    
    def handle_secondary_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.secondary_frame_signal.emit(qt_image)
    
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

    def add_frame_overlays(self, frame, width, height):
        """Add informational overlays to frame"""
        try:
            # Count display
            cv2.putText(frame, f"Count: {self.count_value}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Target count info
            if self.target_count > 0:
                target_text = f"Target: {self.target_count}"
                cv2.putText(frame, target_text, (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                # Progress indicator
                progress = min(1.0, self.count_value / self.target_count)
                bar_width = 200
                bar_height = 20
                bar_x = width - bar_width - 20
                bar_y = 120
                
                # Background bar
                cv2.rectangle(frame, (bar_x, bar_y), 
                            (bar_x + bar_width, bar_y + bar_height), 
                            (100, 100, 100), -1)
                
                # Progress bar
                progress_width = int(bar_width * progress)
                if progress_width > 0:
                    color = (0, 165, 255) if self.target_reached else (0, 255, 0)
                    cv2.rectangle(frame, (bar_x, bar_y), 
                                (bar_x + progress_width, bar_y + bar_height), 
                                color, -1)
                
                # Progress text
                progress_text = f"{self.count_value}/{self.target_count} ({progress*100:.1f}%)"
                cv2.putText(frame, progress_text, (bar_x, bar_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Status indicators
            recording_status = "REC" if self.recording else ""
            cv2.putText(frame, recording_status, (width-80, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            camera_indicator = f"Active: {self.active_camera.upper()}"
            cv2.putText(frame, camera_indicator, (width-200, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Mode info
            cv2.putText(frame,
                        f"Edit:{'ON' if self.edit_mode else 'OFF'}  Axis:{self.axis}",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
        except Exception as e:
            print(f"Error adding overlays: {e}")
    
    def run(self):
        """Enhanced main loop with comprehensive error handling"""
        self.run_flag = True
        
        try:
            if not self.init_video(show_connection_status=False):
                self.run_flag = False
                return
            
            if not self.init_model():
                self.run_flag = False
                return
            
            self.load_config()
            
            frame_count = 0
            error_count = 0
            max_errors = 10
            
            while self.run_flag and error_count < max_errors:
                try:
                    if not self.paused and self.cap_thread is not None:
                        # Memory management check
                        if frame_count % 300 == 0:
                            self.check_memory_usage()
                        
                        primary_frame = self.cap_thread.get_frame()
                        secondary_frame = None
                        
                        if self.secondary_cap_thread is not None:
                            secondary_frame = self.secondary_cap_thread.get_frame()
                        
                        if primary_frame is None:
                            self.msleep(10)
                            continue
                        
                        # Process frame
                        H, W = primary_frame.shape[:2]
                        primary_display_frame = primary_frame.copy()
                            
                        # Process detection and tracking if not in edit mode
                        if not self.edit_mode and self.roi is not None:
                            detection_frame = primary_frame if self.active_camera == "primary" else secondary_frame

                            if detection_frame is not None:
                                results = self.model.track(
                                    source=detection_frame,
                                    persist=True,
                                    tracker=self.tracker,
                                    conf=self.conf_thresh,
                                    iou=self.iou_thresh,
                                    verbose=False
                                )
                                    
                                if len(results) > 0:
                                    r = results[0]
                                    if r.boxes is not None and r.boxes.id is not None:
                                        ids = r.boxes.id.cpu().tolist()
                                        xyxy = r.boxes.xyxy.cpu().tolist()
                                        cls = r.boxes.cls.cpu().tolist() if r.boxes.cls is not None else [None] * len(ids)
                                        
                                        for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, cls):
                                            if self.class_filter is not None and (c is None or int(c) not in self.class_filter):
                                                continue
                                            
                                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                            
                                            self.update_counter(int(tid), (cx, cy))
                                            
                                            cv2.rectangle(primary_display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            cv2.circle(primary_display_frame, (cx, cy), 3, (0, 0, 255), -1)
                                            cv2.putText(primary_display_frame, f"ID {int(tid)}", (x1, y1 - 6),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                        
                        # Draw ROI if it exists
                        if self.roi is not None:
                            primary_display_frame = self.draw_roi(primary_display_frame, self.roi)
                        
                        # Add overlays
                        self.add_frame_overlays(primary_display_frame, W, H)
                        
                        # Write frames to recordings
                        self.write_frame_to_recording(primary_display_frame, secondary_frame)
                        
                        # Convert to QImage for display
                        rgb_image = cv2.cvtColor(primary_display_frame, cv2.COLOR_BGR2RGB)
                        h, w, ch = rgb_image.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                        self.change_pixmap_signal.emit(qt_image)
                        
                        frame_count += 1
                        
                        # Periodic count publishing
                        current_time = time.time()
                        if current_time - self.last_count_publish_time >= self.count_publish_interval:
                            self.last_count_publish_time = current_time
                            self.update_count_signal.emit(self.count_value)
                    
                    self.msleep(10)
                    
                except Exception as e:
                    error_count += 1
                    error_msg = f"Frame processing error #{error_count}: {str(e)}"
                    self.update_info_signal.emit(error_msg)
                    print(f"ERROR: {error_msg}")
                    print(traceback.format_exc())
                    
                    if error_count >= max_errors:
                        self.update_info_signal.emit("❌ Too many errors, stopping thread")
                        break
                    
                    self.msleep(100)
                    
        except Exception as e:
            error_msg = f"Critical error in video thread: {str(e)}"
            self.update_info_signal.emit(error_msg)
            print(f"CRITICAL ERROR: {error_msg}")
            print(traceback.format_exc())
        finally:
            self.safe_cleanup()

    def stop(self):
        """Enhanced stop method"""
        self.run_flag = False
        self.safe_cleanup()
        self.wait(5000)

    def toggle_pause(self):
        self.paused = not self.paused

    def reset_counter(self):
        self.count_value = 0
        with self.state_lock:
            self.obj_state.clear()
        self.target_reached = False
        self.update_count_signal.emit(self.count_value)
    
    def clear_roi(self):
        self.roi = None
        with self.state_lock:
            self.obj_state.clear()
        self.count_value = 0
        self.target_reached = False
        self.update_count_signal.emit(self.count_value)

# =========================
# Enhanced Main GUI Window
# =========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual Camera Belt Counter - Object Tracking System")
        
        # Enhanced state management
        self.last_sent_count = -1
        self.video_path = ""
        self.start_on_video_load = False
        self.target_reached_handled = False
        self.application_closing = False
        self._last_mqtt_count_time = 0
        
        # Get screen dimensions
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        # Set window size based on screen dimensions
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
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
        left_scroll.setMinimumWidth(300)
        left_scroll.setMaximumWidth(int(window_width * 0.4))
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        # Camera selection section
        camera_group = QGroupBox("📷 Dual Camera Setup")
        camera_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        camera_layout = QVBoxLayout(camera_group)
        
        # Primary camera
        camera_layout.addWidget(QLabel("Primary Camera:"))
        self.primary_camera_combo = QComboBox()
        self.primary_camera_combo.addItems(list(CAMERA_OPTIONS.keys()))
        self.primary_camera_combo.currentTextChanged.connect(self.on_primary_camera_changed)
        camera_layout.addWidget(self.primary_camera_combo)
        
        self.primary_rtsp_edit = QLineEdit(DEFAULT_VIDEO_SOURCE)
        camera_layout.addWidget(QLabel("Primary RTSP URL:"))
        camera_layout.addWidget(self.primary_rtsp_edit)
        
        self.connect_primary_btn = QPushButton("🔗 Connect Primary")
        self.connect_primary_btn.clicked.connect(self.connect_primary_rtsp)
        camera_layout.addWidget(self.connect_primary_btn)
        
        # Secondary camera
        camera_layout.addWidget(QLabel("Secondary Camera:"))
        self.secondary_camera_combo = QComboBox()
        self.secondary_camera_combo.addItems(list(CAMERA_OPTIONS.keys()))
        self.secondary_camera_combo.currentTextChanged.connect(self.on_secondary_camera_changed)
        camera_layout.addWidget(self.secondary_camera_combo)
        
        self.secondary_rtsp_edit = QLineEdit("")
        camera_layout.addWidget(QLabel("Secondary RTSP URL:"))
        camera_layout.addWidget(self.secondary_rtsp_edit)
        
        self.connect_secondary_btn = QPushButton("🔗 Connect Secondary")
        self.connect_secondary_btn.clicked.connect(self.connect_secondary_rtsp)
        camera_layout.addWidget(self.connect_secondary_btn)
        
        # Active camera selection
        camera_layout.addWidget(QLabel("Active Detection Camera:"))
        self.active_camera_combo = QComboBox()
        self.active_camera_combo.addItems(["Primary Camera", "Secondary Camera"])
        self.active_camera_combo.currentTextChanged.connect(self.on_active_camera_changed)
        camera_layout.addWidget(self.active_camera_combo)
        
        # ROI settings section
        roi_group = QGroupBox("🎯 ROI Settings")
        roi_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        roi_layout = QVBoxLayout(roi_group)
        
        self.draw_roi_btn = QPushButton("📐 Enable ROI Drawing")
        self.draw_roi_btn.clicked.connect(self.toggle_edit_mode)
        
        self.clear_roi_btn = QPushButton("🗑️ Clear ROI")
        self.clear_roi_btn.clicked.connect(self.clear_roi)
        
        roi_layout.addWidget(self.draw_roi_btn)
        roi_layout.addWidget(self.clear_roi_btn)
        
        # Target Count section
        target_group = QGroupBox("🎯 Target Count Settings")
        target_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        target_layout = QVBoxLayout(target_group)
        
        target_layout.addWidget(QLabel("Target Count:"))
        self.target_count_spin = QSpinBox()
        self.target_count_spin.setRange(0, 10000)
        self.target_count_spin.setValue(0)
        self.target_count_spin.valueChanged.connect(self.on_target_count_changed)
        target_layout.addWidget(self.target_count_spin)
        
        self.set_target_btn = QPushButton("🎯 Set Target Count")
        self.set_target_btn.clicked.connect(self.set_target_count)
        target_layout.addWidget(self.set_target_btn)
        
        self.target_status_label = QLabel("Target: Not Set")
        self.target_status_label.setStyleSheet("font-weight: bold; color: #FFA500;")
        target_layout.addWidget(self.target_status_label)
        
        # Recording settings section
        recording_group = QGroupBox("🎥 Recording Settings")
        recording_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
        mqtt_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
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
        control_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        control_layout = QVBoxLayout(control_group)
        
        self.start_btn = QPushButton("▶️ Start Counting & Recording")
        self.start_btn.clicked.connect(self.start_counting)
        
        self.stop_btn = QPushButton("⏹️ Stop Counting & Recording")
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
        status_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        status_layout = QVBoxLayout(status_group)
        
        self.count_label = QLabel("Count: 0")
        self.count_label.setStyleSheet("font-weight: bold; font-size: 60px; color: #FF6B6B;")
        self.info_label = QLabel("Ready to start")
        self.info_label.setWordWrap(True)
        
        status_layout.addWidget(self.count_label)
        status_layout.addWidget(self.info_label)
        
        # Add all groups to left layout
        left_layout.addWidget(camera_group)
        left_layout.addWidget(roi_group)
        left_layout.addWidget(status_group)
        left_layout.addWidget(target_group)
        left_layout.addWidget(recording_group)
        left_layout.addWidget(control_group)
        left_layout.addWidget(mqtt_group)
        left_layout.addStretch()
        
        # Set left panel to scroll area
        left_scroll.setWidget(left_panel)
        
        # Right panel for video display
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setSpacing(10)
        video_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create a vertical splitter for the two camera views
        video_splitter = QSplitter(Qt.Orientation.Vertical)
        video_splitter.setChildrenCollapsible(False)
        
        # Primary camera (top)
        primary_video_group = QGroupBox("📹 Primary Camera (Detection Active)")
        primary_video_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        primary_video_layout = QVBoxLayout(primary_video_group)
        
        self.primary_video_label = QLabel()
        self.primary_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.primary_video_label.setMinimumSize(320, 240)
        self.primary_video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.primary_video_label.setText("Primary Camera Feed")
        self.primary_video_label.setStyleSheet("""
            border: 2px solid #4A4A4A; 
            background-color: #1E1E1E; 
            border-radius: 5px;
            color: #FFFFFF;
            font-weight: bold;
        """)
        primary_video_layout.addWidget(self.primary_video_label)
        
        # Secondary camera (bottom)
        secondary_video_group = QGroupBox("📹 Secondary Camera")
        secondary_video_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        secondary_video_layout = QVBoxLayout(secondary_video_group)
        
        self.secondary_video_label = QLabel()
        self.secondary_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.secondary_video_label.setMinimumSize(320, 240)
        self.secondary_video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.secondary_video_label.setText("Secondary Camera Feed")
        self.secondary_video_label.setStyleSheet("""
            border: 2px solid #4A4A4A; 
            background-color: #1E1E1E; 
            border-radius: 5px;
            color: #FFFFFF;
            font-weight: bold;
        """)
        secondary_video_layout.addWidget(self.secondary_video_label)
        
        # Add both video groups to splitter
        video_splitter.addWidget(primary_video_group)
        video_splitter.addWidget(secondary_video_group)
        
        # Set initial splitter sizes (50/50)
        video_splitter.setSizes([int(window_height * 0.5), int(window_height * 0.5)])
        
        # Add splitter to video layout
        video_layout.addWidget(video_splitter)
        
        # Add panels to main splitter
        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(video_container)
        
        # Set main splitter properties
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        
        # Set splitter sizes (30% for controls, 70% for video)
        splitter_sizes = [int(window_width * 0.3), int(window_width * 0.7)]
        main_splitter.setSizes(splitter_sizes)
        
        # Enhanced initialization
        self.enhanced_init()
        
        # Set initial UI state
        self.on_primary_camera_changed("Camera 1")
        self.on_secondary_camera_changed("Camera 2")
        
        # Apply styles
        self.apply_styles()

    def enhanced_init(self):
        """Enhanced initialization with better error handling"""
        try:
            # Initialize video thread
            self.thread = VideoThread()
            self.thread.change_pixmap_signal.connect(self.update_primary_image)
            self.thread.secondary_frame_signal.connect(self.update_secondary_image)
            self.thread.update_count_signal.connect(self.update_count)
            self.thread.update_info_signal.connect(self.update_info)
            self.thread.connection_status_signal.connect(self.handle_connection_status)
            self.thread.recording_status_signal.connect(self.handle_recording_status)
            self.thread.target_reached_signal.connect(self.handle_target_reached)
            self.thread.counting_status_signal.connect(self.handle_counting_status)
            self.thread.memory_warning_signal.connect(self.handle_memory_warning)
            
            # Initialize MQTT client
            self.mqtt_client = mqtt.Client()
            self.mqtt_client.on_connect = self.on_mqtt_connect
            self.mqtt_client.on_message = self.on_mqtt_message
            
            # Start MQTT with error handling
            self.safe_start_mqtt()
            
        except Exception as e:
            self.show_critical_error(f"Application initialization failed: {e}")

    def safe_start_mqtt(self):
        """Safely start MQTT client"""
        try:
            mqtt_thread = threading.Thread(target=self.run_mqtt, daemon=True)
            mqtt_thread.start()
            self.update_info("MQTT client starting...")
        except Exception as e:
            self.update_info(f"MQTT thread start failed: {e}")

    @pyqtSlot(float)
    def handle_memory_warning(self, memory_mb):
        """Handle memory usage warnings"""
        try:
            warning_msg = f"🚨 High memory usage detected: {memory_mb:.1f}MB"
            self.update_info(warning_msg)
            
            QTimer.singleShot(0, lambda: self.show_memory_warning(memory_mb))
            
        except Exception as e:
            print(f"Error handling memory warning: {e}")

    def show_memory_warning(self, memory_mb):
        """Show non-blocking memory warning"""
        try:
            msg = QMessageBox(self)
            msg.setWindowTitle("Memory Warning")
            msg.setText(f"High memory usage: {memory_mb:.1f}MB\n\nApplication may become unstable.")
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setWindowModality(Qt.WindowModality.NonModal)
            msg.show()
        except Exception as e:
            print(f"Error showing memory warning: {e}")

    def apply_styles(self):
        # ... (keep the existing apply_styles method as is) ...
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_height = screen_geometry.height()
        
        base_font_size = 9 if screen_height > 1080 else 8
        
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
            
            QSplitter::handle {{
                background-color: #4A4A4A;
            }}
            
            QSplitter::handle:horizontal {{
                width: 4px;
            }}
            
            QSplitter::handle:vertical {{
                height: 4px;
            }}
        """)
        
        # Special button styling
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
        
        self.connect_primary_btn.setStyleSheet("""
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
        
        self.connect_secondary_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-weight: bold;
                border: 1px solid #7B1FA2;
            }
            QPushButton:hover {
                background-color: #8E24AA;
            }
            QPushButton:pressed {
                background-color: #7B1FA2;
            }
        """)
        
        self.set_target_btn.setStyleSheet("""
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

    @pyqtSlot(bool)
    def handle_counting_status(self, is_counting):
        """Update UI based on counting status"""
        if is_counting:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        else:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    def on_target_count_changed(self, value):
        """Update target status label when spinbox value changes"""
        if value > 0:
            self.target_status_label.setText(f"Target: {value}")
            self.target_status_label.setStyleSheet("font-weight: bold; color: #FFA500;")
        else:
            self.target_status_label.setText("Target: Not Set")
            self.target_status_label.setStyleSheet("font-weight: bold; color: #FFA500;")

    def set_target_count(self):
        """Set the target count in the video thread"""
        target = self.target_count_spin.value()
        if target > 0:
            self.thread.set_target_count(target)
            self.target_status_label.setText(f"Target: {target}")
            self.target_status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        else:
            self.thread.set_target_count(0)
            self.target_status_label.setText("Target: Not Set")
            self.target_status_label.setStyleSheet("font-weight: bold; color: #FFA500;")

    @pyqtSlot(int, int)
    def handle_target_reached(self, current_count, target_count):
        """Robust target reached handler"""
        try:
            if self.target_reached_handled:
                return
                
            self.target_reached_handled = True
            
            QTimer.singleShot(0, lambda: self._execute_target_reached(current_count, target_count))
            
        except Exception as e:
            error_msg = f"Error in target reached handler: {e}"
            self.update_info(error_msg)
            print(f"TARGET_REACHED_ERROR: {error_msg}")

    def _execute_target_reached(self, current_count, target_count):
        """Execute target reached actions in main thread"""
        try:
            message = f"🎉 TARGET REACHED! Count: {current_count}, Target: {target_count}"
            self.update_info(message)
            
            # Send notifications in separate threads
            threading.Thread(
                target=self.send_target_reached_notification,
                args=(current_count, target_count),
                daemon=True
            ).start()
            
            threading.Thread(
                target=self.send_target_reached_mqtt,
                args=(current_count, target_count),
                daemon=True
            ).start()
            
            # Show popup
            self.show_target_reached_popup(current_count, target_count)
            
        except Exception as e:
            error_msg = f"Error executing target reached: {e}"
            self.update_info(error_msg)
            print(f"EXECUTE_TARGET_ERROR: {error_msg}")
        finally:
            QTimer.singleShot(1000, lambda: setattr(self, 'target_reached_handled', False))

    def send_target_reached_notification(self, current_count, target_count):
        """Send target reached notification with robust error handling"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                payload = {
                    "count": current_count,
                    "target_count": target_count,
                    "loading_dock": "WL 2",
                    "wagon_no": "unknown",
                    "video": getattr(self, 'video_path', 'unknown').split("/")[-1],
                    "status": "target_reached",
                    "message": f"Target count of {target_count} reached with count {current_count}"
                }
                
                response = requests.post(SERVER_URL, data=payload, timeout=10)
                response.raise_for_status()
                
                self.update_info(f"✅ Target notification sent successfully (attempt {attempt + 1})")
                break
                
            except requests.exceptions.Timeout:
                self.update_info(f"⏰ Target notification timeout (attempt {attempt + 1})")
            except requests.exceptions.ConnectionError:
                self.update_info(f"🔌 Target notification connection error (attempt {attempt + 1})")
            except Exception as e:
                self.update_info(f"❌ Target notification error (attempt {attempt + 1}): {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2)

    def send_target_reached_mqtt(self, current_count, target_count):
        """Send target reached MQTT notification safely"""
        try:
            if not hasattr(self, 'mqtt_client') or not self.mqtt_client.is_connected():
                self.update_info("MQTT not connected for target notification")
                return
                
            payload = {
                "current_count": current_count,
                "target_count": target_count,
                "status": "target_reached",
                "timestamp": datetime.now().isoformat(),
                "message": f"Target count of {target_count} has been reached"
            }
            
            result = self.mqtt_client.publish(MQTT_TOPIC_TARGET_REACHED, json.dumps(payload))
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.update_info(f"✅ Target reached MQTT notification sent")
            else:
                self.update_info(f"❌ MQTT publish failed with code: {result.rc}")
                
        except Exception as e:
            self.update_info(f"❌ MQTT target notification error: {e}")

    def show_target_reached_popup(self, current_count, target_count):
        """Show target reached popup safely"""
        try:
            if self.application_closing or not QApplication.instance():
                return
                
            msg = QMessageBox(self)
            msg.setWindowTitle("🎉 Target Reached!")
            msg.setText(f"Target count has been reached!\n\nCurrent Count: {current_count}\nTarget Count: {target_count}")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.setWindowModality(Qt.WindowModality.ApplicationModal)
            msg.exec()
            
        except Exception as e:
            print(f"Error showing target popup: {e}")

    # MQTT Methods
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
                
                if "set_target_value" in data:
                    target_value = data.get("set_target_value")
                    try:
                        target_value = int(target_value)
                        if target_value > 0:
                            self.target_count_spin.setValue(target_value)
                            self.thread.set_target_count(target_value)
                            self.target_status_label.setText(f"Target: {target_value} (Set via MQTT)")
                            self.target_status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
                            self.update_info(f"Target count set to {target_value} via MQTT")
                        else:
                            self.update_info(f"Invalid target value received via MQTT: {target_value}")
                    except ValueError:
                        self.update_info(f"Invalid target value format received via MQTT: {target_value}")
                
                QTimer.singleShot(0, self.start_counting)
                
            elif action == "stop":
                self.update_info("MQTT Action: Stop")
                wagon_no = data.get("wagon_no", "unknown")
                loading_dock = data.get("loading_dock", "WL 2")
                self.send_stop_payload(wagon_no, loading_dock)
                QTimer.singleShot(0, self.stop_counting)
                
            elif action == "reset":
                self.update_info("MQTT Action: Reset")
                QTimer.singleShot(0, self.reset_counter)

        except Exception as e:
            self.update_info(f"MQTT message error: {e}")

    def send_stop_payload(self, wagon_no, loading_dock):
        try:
            payload = {
                "count": self.thread.count_value,
                "loading_dock": loading_dock,
                "wagon_no": wagon_no,
                "video": self.video_path.split("/")[-1] if self.video_path else "unknown",
                "status": "completed"
            }
            response = requests.post(SERVER_URL, data=payload, timeout=5)
            self.update_info(f"Final count sent to server. Response: {response.json()}")
            self.last_sent_count = self.thread.count_value
        except Exception as e:
            self.update_info(f"Error sending final count: {e}")

    def send_current_count_mqtt(self, count):
        """Send current count to MQTT with rate limiting and error handling"""
        try:
            current_time = time.time()
            if hasattr(self, '_last_mqtt_count_time'):
                if current_time - self._last_mqtt_count_time < 2.8:
                    return
            
            self._last_mqtt_count_time = current_time
            
            if not hasattr(self, 'mqtt_client') or self.mqtt_client is None:
                return
                
            if not self.mqtt_client.is_connected():
                self.update_info("MQTT not connected for count update")
                return
            
            payload = {
                "current_count": count,
                "target_count": getattr(self.thread, 'target_count', 0),
                "timestamp": datetime.now().isoformat(),
                "status": "counting"
            }
            
            result = self.mqtt_client.publish(MQTT_TOPIC_CURRENT_COUNT, json.dumps(payload))
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                self.update_info(f"MQTT count publish failed: {result.rc}")
                
        except Exception as e:
            if random.random() < 0.1:
                self.update_info(f"MQTT count error: {e}")

    def upload_count_to_server(self):
        if self.thread.count_value != self.last_sent_count:
            try:
                payload = {
                    "count": self.thread.count_value,
                    "loading_dock": "WL 2",
                    "video": self.video_path.split("/")[-1] if self.video_path else "unknown",
                    "status": "in_progress"
                }
                response = requests.post(SERVER_URL, data=payload, timeout=2)
                self.update_info(f"Count updated on server: {self.thread.count_value}")
                self.last_sent_count = self.thread.count_value
            except Exception as e:
                self.update_info(f"Error updating count: {e}")
    
    def on_primary_camera_changed(self, camera_name):
        if camera_name == "Custom RTSP":
            self.primary_rtsp_edit.setEnabled(True)
            self.primary_rtsp_edit.setText("")
        else:
            self.primary_rtsp_edit.setEnabled(False)
            self.primary_rtsp_edit.setText(CAMERA_OPTIONS[camera_name])
        
        self.connect_primary_btn.setEnabled(True)
    
    def on_secondary_camera_changed(self, camera_name):
        if camera_name == "Custom RTSP":
            self.secondary_rtsp_edit.setEnabled(True)
            self.secondary_rtsp_edit.setText("")
        else:
            self.secondary_rtsp_edit.setEnabled(False)
            self.secondary_rtsp_edit.setText(CAMERA_OPTIONS[camera_name])
        
        self.connect_secondary_btn.setEnabled(True)
    
    def on_active_camera_changed(self, camera_name):
        if camera_name == "Primary Camera":
            self.thread.set_active_camera("primary")
        else:
            self.thread.set_active_camera("secondary")
    
    def connect_primary_rtsp(self):
        """Manual primary RTSP connection test"""
        rtsp_url = self.primary_rtsp_edit.text().strip()
        if rtsp_url:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.release()
                self.show_connection_popup(True, f"Successfully connected to primary RTSP stream:\n{rtsp_url}")
                self.update_info(f"Connected to primary RTSP stream: {rtsp_url}")
                self.thread.video_source = rtsp_url
                
                if self.thread.isRunning():
                    self.thread.stop()
                    self.thread.init_video(show_connection_status=True)
            else:
                self.show_connection_popup(False, f"Failed to connect to primary RTSP stream:\n{rtsp_url}")
                self.update_info(f"Failed to connect to primary RTSP stream: {rtsp_url}")
        else:
            self.show_connection_popup(False, "Please enter a valid primary RTSP URL")
            self.update_info("Please enter a valid primary RTSP URL")
    
    def connect_secondary_rtsp(self):
        """Manual secondary RTSP connection test"""
        rtsp_url = self.secondary_rtsp_edit.text().strip()
        if rtsp_url:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cap.release()
                self.show_connection_popup(True, f"Successfully connected to secondary RTSP stream:\n{rtsp_url}")
                self.update_info(f"Connected to secondary RTSP stream: {rtsp_url}")
                self.thread.secondary_video_source = rtsp_url
                
                if self.thread.isRunning():
                    self.thread.stop()
                    self.thread.init_video(show_connection_status=False)
            else:
                self.show_connection_popup(False, f"Failed to connect to secondary RTSP stream:\n{rtsp_url}")
                self.update_info(f"Failed to connect to secondary RTSP stream: {rtsp_url}")
        else:
            self.show_connection_popup(False, "Please enter a valid secondary RTSP URL")
            self.update_info("Please enter a valid secondary RTSP URL")
    
    def show_connection_popup(self, success, message):
        """Show a popup message for connection status"""
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
        """Handle connection status signals from the video thread"""
        self.show_connection_popup(status, message)
        self.update_info(message)
    
    @pyqtSlot(bool, str)
    def handle_recording_status(self, is_recording, message):
        """Handle recording status updates from video thread"""
        if is_recording:
            self.recording_status_label.setText("Status: Recording...")
            self.recording_status_label.setStyleSheet("color: #FF4444; font-weight: bold;")
        else:
            self.recording_status_label.setText("Status: Ready")
            self.recording_status_label.setStyleSheet("color: #DDDDDD;")
        self.update_info(message)
    
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
            self.update_info("ROI drawing mode enabled. Click and drag on video to draw ROI.")
        else:
            self.draw_roi_btn.setText("📐 Enable ROI Drawing")
            self.draw_roi_btn.setStyleSheet("")
            self.update_info("ROI drawing mode disabled.")
    
    def clear_roi(self):
        self.thread.clear_roi()
        self.update_info("ROI cleared")
    
    def start_counting(self):
        self.thread.video_source = self.primary_rtsp_edit.text()
        self.thread.secondary_video_source = self.secondary_rtsp_edit.text()
        
        # Start recording automatically when counting starts
        self.thread.start_recording()
        
        # Start the thread
        self.thread.start()
        
        # Update UI state
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Set up mouse event handlers
        self.primary_video_label.mousePressEvent = self.mouse_press_event
        self.primary_video_label.mouseMoveEvent = self.mouse_move_event
        self.primary_video_label.mouseReleaseEvent = self.mouse_release_event
        
        self.update_info("Video processing started with recording")
    
    def stop_counting(self):
        """Enhanced stop counting with comprehensive cleanup"""
        try:
            self.update_info("🛑 Stopping counting and recording...")
            
            # Send final count
            self.send_stop_payload("unknown", "WL 2")
            
            # Stop thread safely
            if hasattr(self, 'thread') and self.thread.isRunning():
                self.thread.safe_cleanup()
                if not self.thread.wait(5000):
                    self.update_info("⚠️ Thread didn't stop gracefully, forcing...")
                    self.thread.terminate()
                    self.thread.wait(1000)
            
            # Update UI
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            # Remove mouse event handlers
            self.primary_video_label.mousePressEvent = None
            self.primary_video_label.mouseMoveEvent = None
            self.primary_video_label.mouseReleaseEvent = None
            
            # Force garbage collection
            gc.collect()
            
            self.update_info("✅ Counting stopped safely")
            
        except Exception as e:
            self.update_info(f"❌ Error stopping counting: {e}")
            if hasattr(self, 'thread'):
                self.thread.terminate()
    
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
        if hasattr(self, 'axis_combo'):
            self.axis_combo.setCurrentIndex(0 if self.thread.axis == 'y' else 1)
        if hasattr(self, 'forward_direction_combo'):
            self.forward_direction_combo.setCurrentIndex(0 if self.thread.forward_positive else 1)
        self.target_count_spin.setValue(self.thread.target_count)
        self.on_target_count_changed(self.thread.target_count)
    
    @pyqtSlot(QImage)
    def update_primary_image(self, image):
        self.primary_video_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.primary_video_label.width(), 
            self.primary_video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
    
    @pyqtSlot(QImage)
    def update_secondary_image(self, image):
        self.secondary_video_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.secondary_video_label.width(), 
            self.secondary_video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
    
    @pyqtSlot(int)
    def update_count(self, count):
        self.count_label.setText(f"{count}")
        self.send_current_count_mqtt(count)
    
    @pyqtSlot(str)
    def update_info(self, message):
        self.info_label.setText(message)
        print(f"INFO: {message}")

    def mouse_press_event(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.thread.edit_mode:
            x = event.pos().x()
            y = event.pos().y()
            
            pixmap = self.primary_video_label.pixmap()
            if pixmap:
                label_width = self.primary_video_label.width()
                label_height = self.primary_video_label.height()
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                
                x_offset = (label_width - pixmap_width) // 2 if label_width > pixmap_width else 0
                y_offset = (label_height - pixmap_height) // 2 if label_height > pixmap_height else 0
                
                x = max(0, min(pixmap_width - 1, x - x_offset))
                y = max(0, min(pixmap_height - 1, y - y_offset))
                
                x = int(x * FRAME_WIDTH / pixmap_width)
                y = int(y * FRAME_HEIGHT / pixmap_height)
                
                self.thread.process_mouse_event(
                    cv2.EVENT_LBUTTONDOWN, x, y, 0, FRAME_WIDTH, FRAME_HEIGHT
                )
    
    def mouse_move_event(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.thread.edit_mode:
            x = event.pos().x()
            y = event.pos().y()
            
            pixmap = self.primary_video_label.pixmap()
            if pixmap:
                label_width = self.primary_video_label.width()
                label_height = self.primary_video_label.height()
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                
                x_offset = (label_width - pixmap_width) // 2 if label_width > pixmap_width else 0
                y_offset = (label_height - pixmap_height) // 2 if label_height > pixmap_height else 0
                
                x = max(0, min(pixmap_width - 1, x - x_offset))
                y = max(0, min(pixmap_height - 1, y - y_offset))
                
                x = int(x * FRAME_WIDTH / pixmap_width)
                y = int(y * FRAME_HEIGHT / pixmap_height)
                
                self.thread.process_mouse_event(
                    cv2.EVENT_MOUSEMOVE, x, y, 0, FRAME_WIDTH, FRAME_HEIGHT
                )
    
    def mouse_release_event(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.thread.edit_mode:
            x = event.pos().x()
            y = event.pos().y()
            
            pixmap = self.primary_video_label.pixmap()
            if pixmap:
                label_width = self.primary_video_label.width()
                label_height = self.primary_video_label.height()
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                
                x_offset = (label_width - pixmap_width) // 2 if label_width > pixmap_width else 0
                y_offset = (label_height - pixmap_height) // 2 if label_height > pixmap_height else 0
                
                x = max(0, min(pixmap_width - 1, x - x_offset))
                y = max(0, min(pixmap_height - 1, y - y_offset))
                
                x = int(x * FRAME_WIDTH / pixmap_width)
                y = int(y * FRAME_HEIGHT / pixmap_height)
                
                self.thread.process_mouse_event(
                    cv2.EVENT_LBUTTONUP, x, y, 0, FRAME_WIDTH, FRAME_HEIGHT
                )
    
    def resizeEvent(self, event):
        if hasattr(self, 'thread') and self.thread.isRunning():
            pixmap = self.primary_video_label.pixmap()
            if pixmap and not pixmap.isNull():
                self.primary_video_label.setPixmap(pixmap.scaled(
                    self.primary_video_label.width(),
                    self.primary_video_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
        super().resizeEvent(event)
    
    def closeEvent(self, event):
        """Enhanced close event with guaranteed cleanup"""
        try:
            self.application_closing = True
            
            self.update_info("📋 Application shutting down...")
            
            # Stop counting if running
            if hasattr(self, 'thread') and self.thread.isRunning():
                self.stop_counting()
            
            # Disconnect MQTT
            if hasattr(self, 'mqtt_client'):
                try:
                    self.mqtt_client.disconnect()
                except:
                    pass
            
            # Show recordings info
            self.show_recordings_info()
            
            # Final cleanup
            gc.collect()
            
            self.update_info("👋 Application closed safely")
            
        except Exception as e:
            print(f"Error during close: {e}")
        finally:
            event.accept()

    def show_recordings_info(self):
        """Show information about saved recordings"""
        try:
            if os.path.exists(RECORDINGS_DIR) and os.listdir(RECORDINGS_DIR):
                files = os.listdir(RECORDINGS_DIR)
                recent_files = [f for f in files if f.startswith(('primary_recording_', 'secondary_recording_'))]
                if recent_files:
                    recent_files.sort(key=lambda x: os.path.getmtime(os.path.join(RECORDINGS_DIR, x)), reverse=True)
                    recent_files = recent_files[:4]
                    
                    msg = QMessageBox(self)
                    msg.setWindowTitle("Recordings Saved")
                    msg.setText(f"All recordings have been saved to:\n{os.path.abspath(RECORDINGS_DIR)}\n\nRecent files:\n" + "\n".join(recent_files))
                    msg.setIcon(QMessageBox.Icon.Information)
                    msg.exec()
        except Exception as e:
            print(f"Error showing recordings info: {e}")

    def show_critical_error(self, message):
        """Show critical error message and exit safely"""
        try:
            msg = QMessageBox(self)
            msg.setWindowTitle("Critical Error")
            msg.setText(f"Application encountered a critical error and must close:\n\n{message}")
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
        finally:
            QApplication.quit()

# Global exception handler
def exception_hook(exctype, value, traceback_obj):
    """Global exception handler"""
    error_msg = ''.join(traceback.format_exception(exctype, value, traceback_obj))
    print(f"UNHANDLED EXCEPTION:\n{error_msg}")
    
    try:
        app = QApplication.instance()
        if app is not None:
            msg = QMessageBox()
            msg.setWindowTitle("Unhandled Exception")
            msg.setText(f"An unhandled exception occurred:\n\n{str(value)}")
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
    except:
        pass
    
    sys.__excepthook__(exctype, value, traceback_obj)

sys.excepthook = exception_hook

# =========================
# Application Entry Point
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())