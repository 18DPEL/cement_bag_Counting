import sys
import cv2
import json
import os
import queue
import time
import threading
import requests
from collections import defaultdict, deque
from ultralytics import YOLO
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QComboBox, QLineEdit, QGroupBox, 
                             QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog, QMessageBox,
                             QSizePolicy, QScrollArea, QSplitter, QStackedWidget, QFrame,
                             QTableWidget, QTableWidgetItem, QGridLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
import paho.mqtt.client as mqtt
from datetime import datetime

# =========================
# Configuration
# =========================
DEFAULT_VIDEO_SOURCE = r"rtsp://localhost:8554/mystream"
DEFAULT_MODEL_PATH = r"C:\Users\ayuba\Downloads\best (19).pt"  # Fixed model path
CFG_FILE = "roi_config.json"
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "cement/wagon/control/1"
SERVER_URL = "https://shipeasy.tech/cement/public/api/get_load"
RECORDINGS_DIR = "recordings"  # Directory for saving recordings

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
        self.show_connection_status = show_connection_status  # New parameter
        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480
    
    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
        
        if not cap.isOpened():
            print(f"Error: Could not open video stream {self.video_source}")
            # Only emit signal if we should show connection status
            if self.show_connection_status:
                self.stream_status_signal.emit(False, self.video_source)
            return
        
        # Configure stream properties
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        # Only emit signal if we should show connection status
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
                pass  # Ignore if queue is full
        
        # Main capture loop
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"Frame grab failed for {self.video_source}, retrying...")
                time.sleep(1)
                continue

            # Resize frame to fixed dimensions
            frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            self.new_frame_signal.emit(frame)
            
            # Add to queue for processing
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()  # Remove oldest frame if queue is full
                self.frame_queue.put(frame, timeout=1)
            except queue.Full:
                pass  # Ignore if queue is still full
            except Exception as e:
                print(f"Error adding frame to queue: {e}")

        cap.release()
        # Only emit signal if we should show connection status
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
    update_count_signal = pyqtSignal(int)
    update_info_signal = pyqtSignal(str)
    connection_status_signal = pyqtSignal(bool, str)  # New signal for connection status
    recording_status_signal = pyqtSignal(bool, str)  # Signal for recording status updates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.run_flag = True
        self.paused = False
        self.edit_mode = False  # Changed default to False (disabled)
        
        # Configuration parameters
        self.video_source = DEFAULT_VIDEO_SOURCE
        self.model_path = DEFAULT_MODEL_PATH  # Fixed model path
        self.class_filter = None
        self.conf_thresh = 0.25
        self.iou_thresh = 0.45
        self.tracker = "bytetrack.yaml"
        self.axis = 'y'
        self.forward_positive = True
        
        # ROI and tracking state
        self.roi = None
        self.obj_state = {}
        self.count_value = 0
        
        # Mouse interaction state
        self.dragging = False
        self.resizing = False
        self.drag_offset = (0, 0)
        self.resize_corner = None
        self.mouse_down_pt = None
        self.handle_size = 10
        
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
                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.recording_filename = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.mp4")
                
                # Define codec and create VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    self.recording_filename, 
                    fourcc, 
                    15.0,  # FPS
                    (640, 480)  # Frame size
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
                self.axis = data.get("axis", self.axis)
                self.forward_positive = data.get("forward_positive", self.forward_positive)
                self.update_info_signal.emit(f"Loaded configuration from {CFG_FILE}")
            except Exception as e:
                self.update_info_signal.emit(f"Error loading config: {str(e)}")
    
    def save_config(self):
        try:
            data = {
                "roi": self.roi,
                "axis": self.axis,
                "forward_positive": self.forward_positive
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
        
        # Only draw handles if in edit mode
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
                        delta = -1 if (st['entry_side'] == 'A' and exit_side == 'B') else -1
                    else:
                        delta = +1 if (st['entry_side'] == 'A' and exit_side == 'B') else +1
                    self.count_value += delta
                    self.update_count_signal.emit(self.count_value)
                
                st['state'] = 'outside'
                st['entry_side'] = None
    
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
        if self.cap_thread is not None:
            self.cap_thread.stop()
        
        # Pass the show_connection_status parameter to control when to show messages
        self.cap_thread = FrameCaptureThread(self.video_source, show_connection_status=show_connection_status)
        self.cap_thread.new_frame_signal.connect(self.handle_new_frame)
        self.cap_thread.stream_status_signal.connect(self.handle_stream_status)
        self.cap_thread.start()
        
        return True
    
    def handle_new_frame(self, frame):
        # This method receives frames from the capture thread
        # We'll process them in the main run loop
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
        
        # Don't show connection status during automatic start/stop operations
        if not self.init_video(show_connection_status=False):
            self.run_flag = False
            return
        
        if not self.init_model():
            self.run_flag = False
            return
        
        self.load_config()
        
        while self.run_flag:
            if not self.paused and self.cap_thread is not None:
                # Get frame from queue
                frame = self.cap_thread.get_frame()
                if frame is None:
                    self.msleep(10)  # Sleep briefly if no frame available
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
                                
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.circle(display_frame, (cx, cy), 3, (0, 0, 255), -1)
                                cv2.putText(display_frame, f"ID {int(tid)}", (x1, y1 - 6),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                
                # Draw ROI if it exists
                if self.roi is not None:
                    display_frame = self.draw_roi(display_frame, self.roi)
                
                # Add info text
                cv2.putText(display_frame, f"Count:{self.count_value}", (40, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
                
                # # Add recording indicator
                # recording_status = "REC" if self.recording else ""
                # cv2.putText(display_frame, recording_status, (W-80, 35),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                cv2.putText(display_frame,
                           f"Edit:{'ON' if self.edit_mode else 'OFF'}  Axis:{self.axis}  Forward:+1 is {'A->B' if self.forward_positive else 'B->A'}",
                           (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write the ANNOTATED frame to recording (not the original frame)
                self.write_frame_to_recording(display_frame)
                
                # Convert to QImage for display
                rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                self.change_pixmap_signal.emit(qt_image)
            
            self.msleep(10)  # Small sleep to prevent CPU overuse
        
        # Cleanup
        self.stop_recording()  # Ensure recording is stopped
        if self.cap_thread is not None:
            self.cap_thread.stop()
    
    def stop(self):
        self.run_flag = False
        self.stop_recording()  # Stop recording when thread stops
        if self.cap_thread is not None:
            self.cap_thread.stop()
        self.wait()
    
    def toggle_pause(self):
        self.paused = not self.paused
    
    def reset_counter(self):
        self.count_value = 0
        self.obj_state.clear()
        self.update_count_signal.emit(self.count_value)
    
    def clear_roi(self):
        self.roi = None
        self.obj_state.clear()
        self.count_value = 0
        self.update_count_signal.emit(self.count_value)

# =========================
# Car Detection Window (Modified for Dashboard Integration)
# =========================
class CarDetectionWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_video_thread()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(15)

        # Header
        header = QFrame()
        header.setStyleSheet("background-color: #ffffff; border-radius: 8px;")
        hbox = QHBoxLayout(header)
        hbox.setContentsMargins(15, 10, 15, 10)

        title = QLabel("Live Vehicle Detection & Tracking")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        hbox.addWidget(title)

        hbox.addStretch()
        
        # Control buttons
        self.start_btn = QPushButton("▶️ Start Detection")
        self.stop_btn = QPushButton("⏹️ Stop Detection")
        self.pause_btn = QPushButton("⏸️ Pause")
        self.reset_btn = QPushButton("🔄 Reset Counter")
        
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.reset_btn.clicked.connect(self.reset_counter)
        
        self.stop_btn.setEnabled(False)
        
        hbox.addWidget(self.start_btn)
        hbox.addWidget(self.stop_btn)
        hbox.addWidget(self.pause_btn)
        hbox.addWidget(self.reset_btn)

        layout.addWidget(header)

        # Main content area
        content_layout = QHBoxLayout()
        
        # Video display
        video_frame = QFrame()
        video_frame.setStyleSheet("background-color: #ffffff; border-radius: 10px;")
        video_layout = QVBoxLayout(video_frame)
        
        video_title = QLabel("<b>Live Vehicle Tracking</b>")
        video_title.setStyleSheet("font-size: 16px; padding: 10px;")
        video_layout.addWidget(video_title)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setText("Video feed will appear here")
        self.video_label.setStyleSheet("""
            border: 2px solid #4A4A4A; 
            background-color: #1E1E1E; 
            border-radius: 5px;
            color: #FFFFFF;
            font-weight: bold;
            margin: 10px;
        """)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        video_layout.addWidget(self.video_label)
        
        # Status info
        self.count_label = QLabel("Detected Vehicles: 0")
        self.count_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #0078d7; padding: 10px;")
        video_layout.addWidget(self.count_label)
        
        self.info_label = QLabel("Ready to start vehicle detection")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("padding: 10px;")
        video_layout.addWidget(self.info_label)
        
        content_layout.addWidget(video_frame, 2)
        
        # Controls panel
        controls_frame = QFrame()
        controls_frame.setStyleSheet("background-color: #ffffff; border-radius: 10px;")
        controls_layout = QVBoxLayout(controls_frame)
        
        controls_title = QLabel("<b>Detection Controls</b>")
        controls_title.setStyleSheet("font-size: 16px; padding: 10px;")
        controls_layout.addWidget(controls_title)
        
        # Camera selection
        camera_group = QGroupBox("📷 Camera Selection")
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
        
        # ROI controls
        roi_group = QGroupBox("🎯 ROI Settings")
        roi_layout = QVBoxLayout(roi_group)
        
        self.draw_roi_btn = QPushButton("📐 Enable ROI Drawing")
        self.draw_roi_btn.clicked.connect(self.toggle_edit_mode)
        
        self.clear_roi_btn = QPushButton("🗑️ Clear ROI")
        self.clear_roi_btn.clicked.connect(self.clear_roi)
        
        roi_layout.addWidget(self.draw_roi_btn)
        roi_layout.addWidget(self.clear_roi_btn)
        
        # Recording controls
        recording_group = QGroupBox("🎥 Recording")
        recording_layout = QVBoxLayout(recording_group)
        
        self.recording_status_label = QLabel("Status: Ready")
        self.record_btn = QPushButton("🔴 Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        
        recording_layout.addWidget(self.recording_status_label)
        recording_layout.addWidget(self.record_btn)
        
        controls_layout.addWidget(camera_group)
        controls_layout.addWidget(roi_group)
        controls_layout.addWidget(recording_group)
        controls_layout.addStretch()
        
        content_layout.addWidget(controls_frame, 1)
        layout.addLayout(content_layout)
        
        # Set initial state
        self.on_camera_changed("Camera 1")
        
    def setup_video_thread(self):
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_count_signal.connect(self.update_count)
        self.thread.update_info_signal.connect(self.update_info)
        self.thread.connection_status_signal.connect(self.handle_connection_status)
        self.thread.recording_status_signal.connect(self.handle_recording_status)
        
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
            self.thread.video_source = rtsp_url
            self.update_info(f"RTSP URL set to: {rtsp_url}")
    
    def start_detection(self):
        self.thread.video_source = self.rtsp_edit.text()
        self.thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.video_label.mousePressEvent = self.mouse_press_event
        self.video_label.mouseMoveEvent = self.mouse_move_event
        self.video_label.mouseReleaseEvent = self.mouse_release_event
        
        self.update_info("Vehicle detection started")
    
    def stop_detection(self):
        self.thread.stop_recording()
        self.thread.stop()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        self.video_label.mousePressEvent = None
        self.video_label.mouseMoveEvent = None
        self.video_label.mouseReleaseEvent = None
        
        self.update_info("Vehicle detection stopped")
    
    def toggle_pause(self):
        self.thread.toggle_pause()
        if self.thread.paused:
            self.pause_btn.setText("▶️ Resume")
            self.update_info("Detection paused")
        else:
            self.pause_btn.setText("⏸️ Pause")
            self.update_info("Detection resumed")
    
    def reset_counter(self):
        self.thread.reset_counter()
        self.update_info("Counter reset")
    
    def toggle_edit_mode(self):
        self.thread.edit_mode = not self.thread.edit_mode
        if self.thread.edit_mode:
            self.draw_roi_btn.setText("🛑 Disable ROI Drawing")
            self.update_info("ROI drawing mode enabled")
        else:
            self.draw_roi_btn.setText("📐 Enable ROI Drawing")
            self.update_info("ROI drawing mode disabled")
    
    def clear_roi(self):
        self.thread.clear_roi()
        self.update_info("ROI cleared")
    
    def toggle_recording(self):
        if self.thread.recording:
            self.thread.stop_recording()
            self.record_btn.setText("🔴 Start Recording")
        else:
            self.thread.start_recording()
            self.record_btn.setText("⏹️ Stop Recording")
    
    @pyqtSlot(QImage)
    def update_image(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.video_label.width(), 
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
    
    @pyqtSlot(int)
    def update_count(self, count):
        self.count_label.setText(f"Detected Vehicles: {count}")
    
    @pyqtSlot(str)
    def update_info(self, message):
        self.info_label.setText(message)
    
    @pyqtSlot(bool, str)
    def handle_connection_status(self, status, message):
        self.update_info(message)
    
    @pyqtSlot(bool, str)
    def handle_recording_status(self, is_recording, message):
        if is_recording:
            self.recording_status_label.setText("Status: Recording...")
            self.recording_status_label.setStyleSheet("color: #FF4444; font-weight: bold;")
        else:
            self.recording_status_label.setText("Status: Ready")
            self.recording_status_label.setStyleSheet("color: #333333;")
        self.update_info(message)
    
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

# =========================
# Dashboard with Integrated Video
# =========================
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ShipNexus Dashboard")
        self.resize(1300, 800)
        self.setStyleSheet("""
            QWidget {
                background-color: #f4f6fb;
                font-family: 'Segoe UI';
            }
            QLabel {
                color: #1b1b1b;
            }
        """)

        # Initialize stacked widget for different sections
        self.stacked_widget = QStackedWidget()
        
        # Main container
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # Sidebar + Main Content
        main_layout.addWidget(self.create_sidebar(), 1)
        main_layout.addWidget(self.stacked_widget, 12)  # Use stacked widget instead of main content

        self.setCentralWidget(main_widget)
        
        # Create different sections
        self.create_sections()
        
        # Show dashboard by default
        self.show_dashboard()

    def create_sections(self):
        """Create all application sections"""
        # Dashboard section
        self.dashboard_section = self.create_main_content()
        self.stacked_widget.addWidget(self.dashboard_section)
        
        # Car Detection section
        self.car_detection_section = CarDetectionWindow()
        self.stacked_widget.addWidget(self.car_detection_section)
        
        # Placeholder sections for other features
        self.placeholder_sections = []
        for i in range(3):  # For Fleet Management, Warehouses, Reports
            placeholder = QLabel(f"<h1>Section {i+2} - Coming Soon</h1>")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("background-color: white; border-radius: 10px;")
            self.stacked_widget.addWidget(placeholder)
            self.placeholder_sections.append(placeholder)

    def create_sidebar(self):
        sidebar = QFrame()
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border-right: 1px solid #d9d9d9;
            }
            QPushButton {
                background: transparent;
                text-align: left;
                padding: 10px 20px;
                font-size: 15px;
                color: #333;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #e6f0ff;
                color: #0078d7;
            }
            QPushButton:pressed {
                background-color: #cce0ff;
            }
        """)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        title = QLabel("🚚 ShipNexus")
        title.setStyleSheet("font-weight: bold; font-size: 20px; margin: 10px;")
        layout.addWidget(title)

        # Create navigation buttons
        self.dashboard_btn = QPushButton("Dashboard")
        self.car_detection_btn = QPushButton("Car Detection")
        self.fleet_btn = QPushButton("Fleet Management")
        self.warehouse_btn = QPushButton("Warehouses")
        self.reports_btn = QPushButton("Reports")

        # Connect buttons to navigation functions
        self.dashboard_btn.clicked.connect(self.show_dashboard)
        self.car_detection_btn.clicked.connect(self.show_car_detection)
        self.fleet_btn.clicked.connect(lambda: self.show_section(2))
        self.warehouse_btn.clicked.connect(lambda: self.show_section(3))
        self.reports_btn.clicked.connect(lambda: self.show_section(4))

        # Add buttons to layout
        layout.addWidget(self.dashboard_btn)
        layout.addWidget(self.car_detection_btn)
        layout.addWidget(self.fleet_btn)
        layout.addWidget(self.warehouse_btn)
        layout.addWidget(self.reports_btn)

        layout.addStretch()

        layout.addWidget(QPushButton("Settings"))
        layout.addWidget(QPushButton("Logout"))

        return sidebar

    def create_main_content(self):
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(15)

        # Header
        layout.addWidget(self.create_header())

        # Top stats cards
        layout.addLayout(self.create_top_cards())

        # Map + Notifications + Chart row
        mid_layout = QHBoxLayout()
        mid_layout.addWidget(self.create_live_tracking_frame(), 2)  # Updated to live tracking
        mid_layout.addWidget(self.create_notifications_frame(), 1)
        layout.addLayout(mid_layout)

        # Table
        layout.addWidget(self.create_recent_table())

        return content

    def create_header(self):
        header = QFrame()
        header.setStyleSheet("background-color: #ffffff; border-radius: 8px;")
        hbox = QHBoxLayout(header)
        hbox.setContentsMargins(15, 10, 15, 10)

        title = QLabel("Dashboard Overview")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        hbox.addWidget(title)

        hbox.addStretch()
        hbox.addWidget(QLineEdit(placeholderText="Search..."))
        hbox.addWidget(QLabel("10 Feb 2025"))

        return header

    def create_top_cards(self):
        grid = QGridLayout()
        grid.setSpacing(10)

        stats = [
            ("Total Shipments", "7000", "Last week +5%"),
            ("Active Vehicles", "900", "Last week +4%"),
            ("Warehouse Capacity", "48%", "Last week -3%"),
            ("Earnings & Costs", "₹ 3,05,000", "Last week +6%"),
        ]

        for i, (title, value, info) in enumerate(stats):
            frame = QFrame()
            frame.setStyleSheet("""
                QFrame {
                    background-color: #ffffff;
                    border-radius: 10px;
                }
                QLabel {
                    color: #333;
                }
            """)
            vbox = QVBoxLayout(frame)
            vbox.addWidget(QLabel(f"<b>{title}</b>"))
            vbox.addWidget(QLabel(f"<span style='font-size:22px; color:#0078d7;'>{value}</span>"))
            vbox.addWidget(QLabel(f"<span style='color:green;'>{info}</span>"))
            grid.addWidget(frame, 0, i)

        return grid

    def create_live_tracking_frame(self):
        """Create the live vehicle tracking frame with video feed"""
        frame = QFrame()
        frame.setStyleSheet("background-color: #ffffff; border-radius: 10px;")
        layout = QVBoxLayout(frame)
        
        # Header with controls
        header_layout = QHBoxLayout()
        title = QLabel("<b>Live Vehicles Tracking</b>")
        title.setStyleSheet("font-size: 16px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Quick action buttons
        self.quick_start_btn = QPushButton("▶️ Start")
        self.quick_stop_btn = QPushButton("⏹️ Stop")
        self.quick_reset_btn = QPushButton("🔄 Reset")
        
        self.quick_start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
        """)
        
        self.quick_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #F44336;
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #E53935;
            }
        """)
        
        self.quick_reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #FB8C00;
            }
        """)
        
        self.quick_stop_btn.setEnabled(False)
        
        header_layout.addWidget(self.quick_start_btn)
        header_layout.addWidget(self.quick_stop_btn)
        header_layout.addWidget(self.quick_reset_btn)
        
        layout.addLayout(header_layout)
        
        # Video display area
        self.dashboard_video_label = QLabel()
        self.dashboard_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dashboard_video_label.setMinimumSize(640, 360)
        self.dashboard_video_label.setText("Live vehicle tracking feed\nClick 'Start' to begin detection")
        self.dashboard_video_label.setStyleSheet("""
            border: 2px solid #4A4A4A; 
            background-color: #1E1E1E; 
            border-radius: 5px;
            color: #FFFFFF;
            font-weight: bold;
            margin: 10px;
        """)
        self.dashboard_video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        layout.addWidget(self.dashboard_video_label)
        
        # Status information
        status_layout = QHBoxLayout()
        
        self.vehicle_count_label = QLabel("Vehicles Detected: 0")
        self.vehicle_count_label.setStyleSheet("font-weight: bold; font-size: 14px; color: #0078d7;")
        
        self.tracking_status_label = QLabel("Status: Ready")
        self.tracking_status_label.setStyleSheet("font-size: 12px; color: #666;")
        
        status_layout.addWidget(self.vehicle_count_label)
        status_layout.addStretch()
        status_layout.addWidget(self.tracking_status_label)
        
        layout.addLayout(status_layout)
        
        # Setup video thread for dashboard
        self.setup_dashboard_video()
        
        # Connect quick action buttons
        self.quick_start_btn.clicked.connect(self.quick_start_detection)
        self.quick_stop_btn.clicked.connect(self.quick_stop_detection)
        self.quick_reset_btn.clicked.connect(self.quick_reset_counter)
        
        return frame

    def setup_dashboard_video(self):
        """Setup video thread for dashboard live tracking"""
        self.dashboard_thread = VideoThread()
        self.dashboard_thread.change_pixmap_signal.connect(self.update_dashboard_image)
        self.dashboard_thread.update_count_signal.connect(self.update_dashboard_count)
        self.dashboard_thread.update_info_signal.connect(self.update_dashboard_status)

    def quick_start_detection(self):
        """Quick start detection from dashboard"""
        self.dashboard_thread.start()
        self.quick_start_btn.setEnabled(False)
        self.quick_stop_btn.setEnabled(True)
        self.tracking_status_label.setText("Status: Tracking Active")
        self.tracking_status_label.setStyleSheet("font-size: 12px; color: #4CAF50; font-weight: bold;")

    def quick_stop_detection(self):
        """Quick stop detection from dashboard"""
        self.dashboard_thread.stop()
        self.quick_start_btn.setEnabled(True)
        self.quick_stop_btn.setEnabled(False)
        self.tracking_status_label.setText("Status: Stopped")
        self.tracking_status_label.setStyleSheet("font-size: 12px; color: #F44336;")

    def quick_reset_counter(self):
        """Quick reset counter from dashboard"""
        self.dashboard_thread.reset_counter()

    @pyqtSlot(QImage)
    def update_dashboard_image(self, image):
        """Update dashboard video display"""
        self.dashboard_video_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.dashboard_video_label.width(), 
            self.dashboard_video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    @pyqtSlot(int)
    def update_dashboard_count(self, count):
        """Update vehicle count in dashboard"""
        self.vehicle_count_label.setText(f"Vehicles Detected: {count}")

    @pyqtSlot(str)
    def update_dashboard_status(self, message):
        """Update status in dashboard"""
        self.tracking_status_label.setText(f"Status: {message}")

    def create_notifications_frame(self):
        frame = QFrame()
        frame.setStyleSheet("background-color: #ffffff; border-radius: 10px;")
        layout = QVBoxLayout(frame)
        
        title = QLabel("<b>Alerts & Notifications</b>")
        title.setStyleSheet("font-size: 16px; padding: 10px;")
        layout.addWidget(title)
        
        # Vehicle detection alerts
        detection_alerts = QGroupBox("🚗 Vehicle Detection")
        detection_alerts.setStyleSheet("""
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        detection_layout = QVBoxLayout(detection_alerts)
        
        alerts = [
            ("✅", "System Ready", "Vehicle detection system is operational"),
            ("📹", "Camera Active", "Live feed from Camera 1"),
            ("🔍", "Tracking Active", "Real-time object tracking enabled"),
            ("📊", "Statistics", "15 vehicles processed today")
        ]
        
        for icon, title_text, desc in alerts:
            alert_widget = QWidget()
            alert_layout = QHBoxLayout(alert_widget)
            alert_layout.setContentsMargins(5, 5, 5, 5)
            
            icon_label = QLabel(icon)
            icon_label.setStyleSheet("font-size: 16px;")
            
            text_widget = QWidget()
            text_layout = QVBoxLayout(text_widget)
            text_layout.setContentsMargins(0, 0, 0, 0)
            
            title_label = QLabel(title_text)
            title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
            
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("font-size: 10px; color: #666;")
            
            text_layout.addWidget(title_label)
            text_layout.addWidget(desc_label)
            
            alert_layout.addWidget(icon_label)
            alert_layout.addWidget(text_widget)
            alert_layout.addStretch()
            
            detection_layout.addWidget(alert_widget)
        
        layout.addWidget(detection_alerts)
        
        # System alerts
        system_alerts = QGroupBox("⚠️ System Alerts")
        system_alerts.setStyleSheet("""
            QGroupBox {
                border: 1px solid #ff6b6b;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ff6b6b;
            }
        """)
        system_layout = QVBoxLayout(system_alerts)
        
        system_alerts_list = [
            ("🚨", "Storage Warning", "Recording storage at 85% capacity"),
            ("⚠️", "Connection Alert", "Backup camera connection unstable"),
            ("🔧", "Maintenance Due", "System maintenance scheduled tomorrow")
        ]
        
        for icon, title_text, desc in system_alerts_list:
            alert_widget = QWidget()
            alert_layout = QHBoxLayout(alert_widget)
            alert_layout.setContentsMargins(5, 5, 5, 5)
            
            icon_label = QLabel(icon)
            icon_label.setStyleSheet("font-size: 16px;")
            
            text_widget = QWidget()
            text_layout = QVBoxLayout(text_widget)
            text_layout.setContentsMargins(0, 0, 0, 0)
            
            title_label = QLabel(title_text)
            title_label.setStyleSheet("font-weight: bold; font-size: 12px; color: #ff6b6b;")
            
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("font-size: 10px; color: #666;")
            
            text_layout.addWidget(title_label)
            text_layout.addWidget(desc_label)
            
            alert_layout.addWidget(icon_label)
            alert_layout.addWidget(text_widget)
            alert_layout.addStretch()
            
            system_layout.addWidget(alert_widget)
        
        layout.addWidget(system_alerts)
        layout.addStretch()
        
        return frame

    def create_recent_table(self):
        frame = QFrame()
        frame.setStyleSheet("background-color: #ffffff; border-radius: 10px;")
        layout = QVBoxLayout(frame)
        layout.addWidget(QLabel("<b>Recent Activities</b>"))

        table = QTableWidget(5, 4)
        table.setHorizontalHeaderLabels(["Time", "Vehicle ID", "Activity", "Status"])
        data = [
            ["14:30:25", "VH-001", "Entered Warehouse", "✅ Tracked"],
            ["14:28:10", "VH-045", "Exited Loading Bay", "✅ Tracked"],
            ["14:25:55", "VH-012", "Parked at Dock 2", "✅ Tracked"],
            ["14:23:40", "VH-078", "In Transit", "🔄 Tracking"],
            ["14:20:15", "VH-033", "Maintenance Alert", "⚠️ Attention"]
        ]
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                table.setItem(i, j, QTableWidgetItem(val))

        layout.addWidget(table)
        return frame

    def show_dashboard(self):
        """Show the dashboard section"""
        self.stacked_widget.setCurrentIndex(0)
        self.update_navigation_style("dashboard")

    def show_car_detection(self):
        """Show the car detection section"""
        self.stacked_widget.setCurrentIndex(1)
        self.update_navigation_style("car_detection")

    def show_section(self, index):
        """Show generic section by index"""
        self.stacked_widget.setCurrentIndex(index)
        self.update_navigation_style(f"section_{index}")

    def update_navigation_style(self, active_section):
        """Update the style of navigation buttons to show active state"""
        # Reset all buttons
        buttons = {
            "dashboard": self.dashboard_btn,
            "car_detection": self.car_detection_btn,
            "section_2": self.fleet_btn,
            "section_3": self.warehouse_btn,
            "section_4": self.reports_btn
        }
        
        for section, button in buttons.items():
            if section == active_section:
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #0078d7;
                        color: white;
                        text-align: left;
                        padding: 10px 20px;
                        font-size: 15px;
                        border-radius: 8px;
                    }
                """)
            else:
                button.setStyleSheet("""
                    QPushButton {
                        background: transparent;
                        text-align: left;
                        padding: 10px 20px;
                        font-size: 15px;
                        color: #333;
                        border-radius: 8px;
                    }
                    QPushButton:hover {
                        background-color: #e6f0ff;
                        color: #0078d7;
                    }
                """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Dashboard()
    window.show()
    sys.exit(app.exec())