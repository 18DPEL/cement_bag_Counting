
import sys
import cv2
import numpy as np
import threading
from queue import Queue, Empty
from ultralytics import YOLO
from collections import defaultdict
import time
import torch
import datetime
import json
import os
import paho.mqtt.client as mqtt
import requests
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QComboBox, QLineEdit, QGroupBox, QMessageBox,
                             QFrame)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint
from PyQt6.QtGui import QIcon,QImage, QPixmap, QKeyEvent, QPainter, QPen, QColor, QFont, QPalette

# Add this at the beginning to ensure the root path is in sys.path
def get_base_path():
    """Get the base path for both development and PyInstaller builds"""
    if getattr(sys, 'frozen', False):
        # Running in a bundle (PyInstaller)
        return sys._MEIPASS
    else:
        # Running in development
        return os.path.dirname(os.path.abspath(__file__))  # only one dirname

# Add the base path to Python path
sys.path.insert(0, get_base_path())

def get_model_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = get_base_path()
    return os.path.join(base_path, relative_path)
class VideoStream(QObject):
    """Class to handle video streaming in a separate thread"""
    new_frame = pyqtSignal(np.ndarray)
    stream_active = pyqtSignal(bool)
    
    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.running = False
        self.frame_queue = Queue(maxsize=2)
        self.last_frame = None  # Store the last good frame
        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480
        
    def start_stream(self):
        self.running = True
        threading.Thread(target=self._capture_frames, daemon=True).start()
        
    def stop_stream(self):
        self.running = False
        
    def _capture_frames(self):
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("Error: Could not open video stream")
            self.stream_active.emit(False)
            return
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        self.stream_active.emit(True)
        
        # Get initial frame immediately
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            self.last_frame = frame
            self.new_frame.emit(frame)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, retrying...")
                time.sleep(1)
                continue

            # Resize frame to fixed dimensions
            frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            self.last_frame = frame
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame, timeout=1)
                self.new_frame.emit(frame)
            except:
                pass

        cap.release()
        self.stream_active.emit(False)
        
    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=1)
        except Empty:
            return None
            
    def get_last_frame(self):
        return self.last_frame


class CementBagOutCounterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(get_model_path('gui/icons/cropped-cropped-logo.ico')))
        self.setWindowTitle("Cement Bag Out Counter Application")
        self.setGeometry(100, 100, 1200, 800)
        # self.setMinimumSize(800, 600)  # Set a reasonable minimum size

        
        # Configuration file path
        self.CONFIG_FILE = "cement_counter_config.json"
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
            }
            QGroupBox {
                background-color: #34495e;
                border: 2px solid #3498db;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                color: #ecf0f1;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QLabel {
                color: #ecf0f1;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1a5276;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
            QComboBox, QLineEdit {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                padding: 5px;
                border-radius: 4px;
                min-height: 25px;
            }
            #countDisplay {
                font-size: 24px;
                font-weight: bold;
                color: #e74c3c;
                background-color: #34495e;
                border: 2px solid #e74c3c;
                border-radius: 5px;
                padding: 10px;
                text-align: center;
            }
        """)
        
        # Frame size constants
        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480
        self.processing_width = self.FRAME_WIDTH
        self.processing_height = self.FRAME_HEIGHT
        
        # Model and tracking variables
        self.model = None
        self.track_history = defaultdict(lambda: [])
        self.count_out = 0  # Only track outgoing count
        self.counted_bags = set()  # Track IDs of bags that have been counted
        self.region_points = []  # Points in original frame coordinates (4 points for quadrilateral)
        self.region_points_scaled = []  # Points in processing frame coordinates
        self.drawing_region = False
        self.region_fixed = False
        self.video_writer = None
        self.is_counting = False
        self.first_frame = None
        self.freeze_for_region_drawing = False  # Changed from freeze_frame
        self.dragging_region = False
        self.dragging_point = None
        self.line_thickness = 3
        self.point_radius = 8
        self.start_on_video_load = False
        self.video_path = None
        self.last_sent_count = -1
        self.has_valid_region = False
        
        # Coordinate mapping variables
        self.original_img_size = None
        self.scaled_pixmap_size = None
        self.pixmap_offset_x = 0
        self.pixmap_offset_y = 0
        self.scale_factor_x = 1.0
        self.scale_factor_y = 1.0
        
        # Video stream
        self.video_stream = None
        self.current_frame = None
        
        # RTSP URLs
        self.rtsp_urls = {
            "Camera 1": "rtsp://admin:Admin%4012345@192.168.53.127:554/cam/realmonitor?channel=1&subtype=0",
            "Camera 2": "rtsp://admin:Admin%4012345@192.168.53.127:554/cam/realmonitor?channel=1&subtype=0",
            "Camera 3": "rtsp://192.168.1.7:8554/mystream"
        }
        
        # MQTT Client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
        self.init_ui()
        self.start_mqtt()
        
        # Load saved region points if available
        self.load_region_points()
            
    def load_region_points(self):
        """Load saved region points from config file if exists"""
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    if 'region_points' in config:
                        self.region_points = [(pt[0], pt[1]) for pt in config['region_points']]
                        self.region_fixed = True
                        self.has_valid_region = True  # Set the valid region flag
                        
                        # Calculate scaled coordinates
                        if self.original_img_size:
                            scale_x = self.processing_width / self.original_img_size[0]
                            scale_y = self.processing_height / self.original_img_size[1]
                            self.region_points_scaled = [
                                (int(pt[0] * scale_x), int(pt[1] * scale_y)) 
                                for pt in self.region_points
                            ]
                            
                        print("Loaded saved region points:", self.region_points)
                        
                        # Enable counting button if we have a valid region
                        if self.has_valid_region:
                            self.start_counting_btn.setEnabled(True)
                            
                        return True
        except Exception as e:
            print("Error loading region points:", e)
        return False
    def save_region_points(self):
        """Save current region points to config file"""
        try:
            config = {
                'region_points': self.region_points,
                'timestamp': datetime.datetime.now().isoformat()
            }
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            print("Saved region points to config file")
        except Exception as e:
            print("Error saving region points:", e)
    
    def init_ui(self):
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel (controls)
        left_panel = QGroupBox("Controls")
        left_panel.setFixedWidth(250)
        left_layout = QVBoxLayout()
        
        # Camera selection
        camera_group = QGroupBox("Camera Configuration")
        camera_layout = QVBoxLayout()
        
        camera_label = QLabel("Select Camera:")
        camera_label.setStyleSheet("font-weight: bold; color: #3498db;")
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(self.rtsp_urls.keys())
        self.camera_combo.setStyleSheet("""
            QComboBox {
                background-color: #ecf0f1;
                selection-background-color: #3498db;
                selection-color: white;
            }
        """)
        
        # Custom RTSP input
        custom_rtsp_label = QLabel("Or enter custom RTSP URL:")
        custom_rtsp_label.setStyleSheet("font-weight: bold; color: #3498db;")
        self.custom_rtsp_input = QLineEdit()
        self.custom_rtsp_input.setPlaceholderText("rtsp://username:password@ip:port/stream")
        
        # Connect button
        self.connect_btn = QPushButton("Connect to RTSP")
        self.connect_btn.clicked.connect(self.connect_to_stream)
        self.connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
            }
            QPushButton:hover {
                background-color: #219653;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
        """)
        
        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(custom_rtsp_label)
        camera_layout.addWidget(self.custom_rtsp_input)
        camera_layout.addWidget(self.connect_btn)
        camera_group.setLayout(camera_layout)
        
        # Region drawing controls
        region_group = QGroupBox("Counting Region Configuration")
        region_layout = QVBoxLayout()
        
        self.region_btn = QPushButton("Draw Counting Region")
        self.region_btn.clicked.connect(self.toggle_region_drawing)
        self.region_btn.setEnabled(False)
        self.region_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
            QPushButton:pressed {
                background-color: #d35400;
            }
        """)
        
        region_layout.addWidget(self.region_btn)
        region_group.setLayout(region_layout)
        
        # Counting controls
        count_group = QGroupBox("Bag Counting Controls")
        count_layout = QVBoxLayout()
        
        self.start_counting_btn = QPushButton("Start Counting")
        self.start_counting_btn.clicked.connect(self.start_counting)
        self.start_counting_btn.setEnabled(False)
        self.start_counting_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
            QPushButton:pressed {
                background-color: #219653;
            }
        """)
        
        self.stop_counting_btn = QPushButton("Stop Counting")
        self.stop_counting_btn.clicked.connect(self.stop_counting)
        self.stop_counting_btn.setEnabled(False)
        self.stop_counting_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        
        # Reset count button
        self.reset_count_btn = QPushButton("Reset Count")
        self.reset_count_btn.clicked.connect(self.reset_counting)
        self.reset_count_btn.setEnabled(True)
        self.reset_count_btn.setStyleSheet("""
            QPushButton {
                background-color: #e67e22;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
        """)
        
        count_layout.addWidget(self.start_counting_btn)
        count_layout.addWidget(self.stop_counting_btn)
        count_layout.addWidget(self.reset_count_btn)
        count_group.setLayout(count_layout)
        
        # Count display
        count_display_group = QGroupBox("Bag Count")
        count_display_layout = QVBoxLayout()
        
        self.count_out_label = QLabel("OUT: 0")
        self.count_out_label.setObjectName("countDisplay")
        self.count_out_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        count_display_layout.addWidget(self.count_out_label)
        count_display_group.setLayout(count_display_layout)
        
        # Add widgets to left layout
        left_layout.addWidget(camera_group)
        left_layout.addWidget(region_group)
        left_layout.addWidget(count_group)
        left_layout.addWidget(count_display_group)
        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        
        # Right panel (video display)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        # Video display frame with border
        video_frame = QFrame()
        video_frame.setFrameShape(QFrame.Shape.StyledPanel)
        video_frame.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border: 3px solid #3498db;
                border-radius: 5px;
            }
        """)
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(5, 5, 5, 5)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(640, 480)
        
        video_layout.addWidget(self.video_label)
        right_layout.addWidget(video_frame)
        right_panel.setLayout(right_layout)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        # Timer for video display
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
        # Load model
        self.load_model()
    
    
    # MQTT Methods
    def start_mqtt(self):
        try:
            mqtt_thread = threading.Thread(target=self.run_mqtt, daemon=True)
            mqtt_thread.start()
        except Exception as e:
            print("MQTT thread start failed:", e)

    def run_mqtt(self):
        try:
            self.mqtt_client.connect("broker.hivemq.com", 1883, 60)
            self.mqtt_client.loop_forever()
        except Exception as e:
            print("MQTT connection failed:", e)

    def on_mqtt_connect(self, client, userdata, flags, rc):
        print("MQTT connected with result code", rc)
        client.subscribe("cement/wagon/control/1")

    def on_mqtt_message(self, client, userdata, msg):
        try:
            print("📩 MQTT Message Received:", msg.payload.decode())
            data = json.loads(msg.payload.decode())
            action = data.get("action")
            print("🔍 Action received:", action)

            if action == "start":
                print("🚀 MQTT Action: Start")

                if not self.video_stream:
                    print("⚠️ No video loaded. Will auto-start after upload.")
                    self.start_on_video_load = True
                    return

                # Don't reset count when starting, continue from previous count
                QTimer.singleShot(0, self.start_counting)

            elif action == "stop":
                print("🛑 MQTT Action: Stop")
                # Get wagon_no and loading_dock from the MQTT message
                wagon_no = data.get("wagon_no", "unknown")
                loading_dock = data.get("loading_dock", "WL 2")  # Default to "WL 2" if not provided
                
                # Send final count with additional information
                self.send_stop_payload(wagon_no, loading_dock)
                
                QTimer.singleShot(0, self.stop_counting)
                
            elif action == "reset":
                print("🔄 MQTT Action: Reset")
                QTimer.singleShot(0, self.reset_counting)

        except Exception as e:
            print("❌ MQTT message error:", e)

    def send_stop_payload(self, wagon_no, loading_dock):
        try:
            payload = {
                "count": self.count_out,
                "loading_dock": loading_dock,
                "wagon_no": wagon_no,
                "video": self.video_path.split("/")[-1] if self.video_path else "unknown",
                "status": "completed"  # Indicate this is a final count
            }
            print("Sending payload:", payload)
            response = requests.post("https://shipeasy.tech/cement/public/api/get_load", 
                                   data=payload,
                                   timeout=1)
            print("Final count sent to server. Response:", response.json())
            self.last_sent_count = self.count_out
        except Exception as e:
            print("Error sending final count:", e)

    def upload_count_to_server(self):
        if self.count_out != getattr(self, 'last_sent_count', -1):
            try:
                payload = {
                    "count": self.count_out,
                    "loading_dock": "WL 2",
                    "video": self.video_path.split("/")[-1] if self.video_path else "unknown",
                    "status": "in_progress"  # Indicate this is an intermediate count
                }
                response = requests.post("https://shipeasy.tech/cement/public/api/get_load", 
                                       data=payload,
                                       timeout=2)
                print("Server response:", response.json())
                self.last_sent_count = self.count_out
            except Exception as e:
                print("Error updating count:", e)
    
    def load_model(self):
        try:
            # Load a model trained to detect cement bags
            model_path = get_model_path(r"C:\Users\ayuba\Downloads\best (15).pt")
            self.model = YOLO(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            
    
    def connect_to_stream(self):
        # Get RTSP URL
        if self.custom_rtsp_input.text():
            rtsp_url = self.custom_rtsp_input.text()
        else:
            selected_camera = self.camera_combo.currentText()
            rtsp_url = self.rtsp_urls[selected_camera]
            
        # Stop any existing stream
        if self.video_stream:
            self.video_stream.stop_stream()
            self.timer.stop()
            
        # Show connecting message
        self.display_connecting_message()
        
        # Start new stream
        self.video_stream = VideoStream(rtsp_url)
        self.video_stream.new_frame.connect(self.handle_new_frame)
        self.video_stream.stream_active.connect(self.handle_stream_status)
        self.video_stream.start_stream()
        
        # Start timer for frame updates
        self.timer.start(30)
        
        # Reset states
        self.reset_region_state()
        
        if self.load_region_points() and self.has_valid_region:
                    # We have a valid saved region, enable counting
                    self.region_btn.setText("Adjust Region")
                    self.start_counting_btn.setEnabled(True)
                    self.show_colored_message(
                        "Success",
                        "RTSP connected successfully. Previously saved counting region loaded.\n"
                        "Click 'Start Counting' to begin or 'Adjust Region' to modify.",
                        QMessageBox.Icon.Information
                    )
        else:
            # No saved region, user needs to draw one
            self.show_colored_message(
                "Instructions",
                "RTSP connected successfully. Click 'Draw Counting Region' to set the counting area.",
                QMessageBox.Icon.Information
            )
        
        # Check if we need to auto-start after connection
        if self.start_on_video_load:
            self.start_on_video_load = False
            QTimer.singleShot(1000, self.start_counting)

        
    def reset_region_state(self):
        """Reset all region-related states"""
        self.region_points = []
        self.region_points_scaled = []
        self.region_fixed = False
        self.drawing_region = False
        self.dragging_region = False
        self.dragging_point = None
        self.freeze_for_region_drawing = True  # Freeze on first frame for region drawing
        self.has_valid_region = False  # Reset the valid region flag
        self.start_counting_btn.setEnabled(False)
        self.region_btn.setEnabled(True)
        self.region_btn.setText("Draw Counting Region")
        self.region_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
            QPushButton:pressed {
                background-color: #d35400;
            }
        """)
        # Remove mouse event handlers
        self.video_label.mousePressEvent = None
        self.video_label.mouseMoveEvent = None
        self.video_label.mouseReleaseEvent = None
        
        # After resetting, check if we have saved points to load
        if self.load_region_points():
            # If points were loaded, enable counting button
            self.region_btn.setText("Adjust Region")
            self.region_btn.setStyleSheet("""
                QPushButton {
                    background-color: #f39c12;
                }
                QPushButton:hover {
                    background-color: #e67e22;
                }
                QPushButton:pressed {
                    background-color: #d35400;
                }
            """)
            self.start_counting_btn.setEnabled(True)
        
    def display_connecting_message(self):
        # Create a black image with connecting message using fixed size
        img = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(img, "Connecting to camera...", 
                   (50, self.FRAME_HEIGHT//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        self.display_image(img)
        
    def show_colored_message(self, title, text, icon=QMessageBox.Icon.Information):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setIcon(icon)
        # Set a more visible background color and text color
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #222244;
                color: #fff;
                font-size: 14px;
                font-weight: bold;
            }
            QLabel {
                color: #fff;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                color: #fff;
                border-radius: 4px;
                min-width: 80px;
                min-height: 28px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #217dbb;
            }
        """)
        msg.exec()

    def handle_stream_status(self, active):
            if active:
                # Connection successful
                self.connect_btn.setEnabled(False)
                self.region_btn.setEnabled(True)
                
                # Get the first frame and store it
                if self.video_stream:
                    self.first_frame = self.video_stream.get_last_frame()
                    if self.first_frame is not None:
                        # Display the first frame frozen
                        self.display_image(self.first_frame)
                        
                        # Check if we have a saved region
                        if self.load_region_points() and self.has_valid_region:
                            # We have a saved region, show it and enable counting
                            display_frame = self.first_frame.copy()
                            for i in range(len(self.region_points)):
                                start_point = self.region_points[i]
                                end_point = self.region_points[(i + 1) % len(self.region_points)]
                                cv2.line(display_frame, start_point, end_point, 
                                        (0, 255, 0), self.line_thickness)
                                cv2.circle(display_frame, start_point, self.point_radius, (0, 255, 0), -1)
                            self.display_image(display_frame)
                            
                            # Enable counting button
                            self.start_counting_btn.setEnabled(True)
                            
                            self.show_colored_message(
                                "Success",
                                "RTSP connected successfully. Previously saved counting region loaded.\n"
                                "Click 'Start Counting' to begin or 'Adjust Region' to modify.",
                                QMessageBox.Icon.Information
                            )
                        else:
                            # No saved region, user needs to draw one
                            self.show_colored_message(
                                "Instructions",
                                "RTSP connected successfully. Click 'Draw Counting Region' to set the counting area.",
                                QMessageBox.Icon.Information
                            )    
    def handle_new_frame(self, frame):
        if self.freeze_for_region_drawing and self.first_frame is None:
            # Store the first frame and freeze for region drawing
            self.first_frame = frame.copy()
        self.current_frame = frame
        
    def update_frame(self):
        if self.current_frame is None:
            return
            # Debug message to check if counting is active
        if self.is_counting:
            print("Counting is ACTIVE - processing frames")
        else:
            print("Counting is INACTIVE - just displaying frames")
                
        # If we're frozen for region drawing
        if self.freeze_for_region_drawing and self.first_frame is not None:
            display_frame = self.first_frame.copy()
            
            # Draw the region if we have points
            if len(self.region_points) > 1:
                # Draw lines between points
                for i in range(len(self.region_points)):
                    start_point = self.region_points[i]
                    end_point = self.region_points[(i + 1) % len(self.region_points)]
                    cv2.line(display_frame, start_point, end_point, 
                             (0, 255, 0) if self.region_fixed else (0, 255, 255), 
                             self.line_thickness)
                
                # Draw endpoints
                for i, pt in enumerate(self.region_points):
                    color = (0, 255, 0) if self.region_fixed else (0, 255, 255)
                    cv2.circle(display_frame, pt, self.point_radius, color, -1)
                    cv2.putText(display_frame, str(i+1), (pt[0]+10, pt[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            self.display_image(display_frame)
            return
            
        # If counting is active, process the frame
        if self.is_counting and self.region_points_scaled and self.region_fixed:
            # Ensure frame is the correct size
            processing_frame = cv2.resize(self.current_frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            
            annotated_frame = self.detect_and_count(
                processing_frame, self.model, self.region_points_scaled, self.track_history
            )
            
            # Display the annotated frame
            self.display_image(annotated_frame)
            
            # Periodically upload count to server
            if time.time() % 15 < 0.1:  # About every 5 seconds
                self.upload_count_to_server()
        else:
            # Just show the current frame with confirmed region if available
            display_frame = self.current_frame.copy()
            if len(self.region_points) > 1:
                color = (0, 255, 0) if self.region_fixed else (0, 255, 255)
                for i in range(len(self.region_points)):
                    start_point = self.region_points[i]
                    end_point = self.region_points[(i + 1) % len(self.region_points)]
                    cv2.line(display_frame, start_point, end_point, color, self.line_thickness)
                for pt in self.region_points:
                    cv2.circle(display_frame, pt, self.point_radius, color, -1)
            self.display_image(display_frame)
    
    def display_image(self, img):
        # Ensure image is the correct size
        img = cv2.resize(img, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        
        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original image size for coordinate mapping
        self.original_img_size = (self.FRAME_WIDTH, self.FRAME_HEIGHT)
        
        # Convert to QImage
        bytes_per_line = 3 * self.FRAME_WIDTH
        q_img = QImage(img_rgb.data, self.FRAME_WIDTH, self.FRAME_HEIGHT, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Create pixmap
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale the pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Store scaling information
        self.scaled_pixmap_size = (scaled_pixmap.width(), scaled_pixmap.height())
        self.scale_factor_x = self.FRAME_WIDTH / scaled_pixmap.width() if scaled_pixmap.width() > 0 else 1.0
        self.scale_factor_y = self.FRAME_HEIGHT / scaled_pixmap.height() if scaled_pixmap.height() > 0 else 1.0
        
        # Calculate offsets for centered display
        self.pixmap_offset_x = (self.video_label.width() - scaled_pixmap.width()) // 2
        self.pixmap_offset_y = (self.video_label.height() - scaled_pixmap.height()) // 2
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def toggle_region_drawing(self):
        if self.first_frame is None:
            # Try to get a new frame if none is available
            if self.video_stream:
                self.first_frame = self.video_stream.get_last_frame()
                if self.first_frame is None:
                    QMessageBox.warning(self, "Warning", "No frame available to draw on")
                    return
            else:
                QMessageBox.warning(self, "Warning", "No video stream connected")
                return
                
        if not self.drawing_region:
            # Start drawing mode
            self.drawing_region = True
            self.region_fixed = False
            self.region_points = []
            self.region_points_scaled = []
            self.has_valid_region = False  # Reset valid region flag
            self.freeze_for_region_drawing = True  # Ensure we're frozen on this frame for region drawing
            
            # Set up mouse event handlers
            self.video_label.mousePressEvent = self.handle_mouse_click
            self.video_label.mouseMoveEvent = self.handle_mouse_move
            self.video_label.mouseReleaseEvent = self.handle_mouse_release
            
            # Update UI
            self.region_btn.setText("Fix Region")
            self.region_btn.setStyleSheet("""
                QPushButton {
                    background-color: #27ae60;
                }
                QPushButton:hover {
                    background-color: #219653;
                }
                QPushButton:pressed {
                    background-color: #1e8449;
                }
            """)
            self.start_counting_btn.setEnabled(False)
            
            # Display the frozen frame
            self.display_image(self.first_frame)
            
            self.show_colored_message(
                "Instructions",
                "Click four points to draw the counting region (quadrilateral).\n"
                "Click points in order (e.g., clockwise or counter-clockwise).\n"
                "Then press 'Fix Region' to confirm.\n"
                "You can drag points to adjust the region."
            )
        else:
            # Check if we have a complete region (4 points)
            if len(self.region_points) != 4:
                self.show_colored_message("Warning", "Please draw a complete region (four points) first", QMessageBox.Icon.Warning)
                return
                
            # Toggle fixed state
            self.region_fixed = not self.region_fixed
            
            if self.region_fixed:
                # Calculate scaled coordinates for processing frame
                if self.original_img_size:
                    scale_x = self.processing_width / self.original_img_size[0]
                    scale_y = self.processing_height / self.original_img_size[1]
                    
                    self.region_points_scaled = [
                        (int(pt[0] * scale_x), int(pt[1] * scale_y)) 
                        for pt in self.region_points
                    ]
                # Set the valid region flag
                self.has_valid_region = True
                # Save the region points
                self.save_region_points()
                
                # Update button
                self.region_btn.setText("Adjust Region")
                self.region_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f39c12;
                    }
                    QPushButton:hover {
                        background-color: #e67e22;
                    }
                    QPushButton:pressed {
                        background-color: #d35400;
                    }
                """)
                self.start_counting_btn.setEnabled(True)
                
                # Unfreeze the frame after region is fixed
                self.freeze_for_region_drawing = False
                
                print(f"Region fixed with points:")
                for i, pt in enumerate(self.region_points):
                    print(f"  Point {i+1}: {pt}")
                print(f"Scaled points:")
                for i, pt in enumerate(self.region_points_scaled):
                    print(f"  Point {i+1}: {pt}")
                
                self.show_colored_message(
                    "Success",
                    f"Counting region fixed!\nClick 'Start Counting' to begin.",
                    QMessageBox.Icon.Information
                )
            else:
                # Region is now adjustable
                self.region_btn.setText("Fix Region")
                self.region_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #27ae60;
                    }
                    QPushButton:hover {
                        background-color: #219653;
                    }
                    QPushButton:pressed {
                        background-color: #1e8449;
                    }
                """)
                self.start_counting_btn.setEnabled(False)
    
    def handle_mouse_click(self, event):
        if not self.drawing_region:
            return
            
        # Get click position relative to the video label
        click_x = event.position().x()
        click_y = event.position().y()
        
        # Convert to image coordinates
        img_x = click_x - self.pixmap_offset_x
        img_y = click_y - self.pixmap_offset_y
        
        # Check if click is within the scaled image bounds
        if (0 <= img_x < self.scaled_pixmap_size[0] and 
            0 <= img_y < self.scaled_pixmap_size[1]):
            
            # Scale to original image coordinates
            orig_x = int(img_x * self.scale_factor_x)
            orig_y = int(img_y * self.scale_factor_y)
            
            # Ensure coordinates are within bounds
            orig_x = max(0, min(orig_x, self.original_img_size[0] - 1))
            orig_y = max(0, min(orig_y, self.original_img_size[1] - 1))
            
            # Check if we're clicking near an existing point
            if len(self.region_points) > 0:
                for i, pt in enumerate(self.region_points):
                    distance = np.sqrt((orig_x - pt[0])**2 + (orig_y - pt[1])**2)
                    if distance < 20:  # Clicked near a point
                        self.dragging_point = i
                        return
                
                # Check if we're clicking near a line (to move entire region)
                if len(self.region_points) == 4:
                    for i in range(4):
                        line_start = self.region_points[i]
                        line_end = self.region_points[(i + 1) % 4]
                        if self.point_to_line_distance((orig_x, orig_y), line_start, line_end) < 10:
                            self.dragging_region = True
                            self.drag_start_pos = (orig_x, orig_y)
                            return
            
            # If we're not dragging, add new points (up to 4)
            if len(self.region_points) < 4:
                self.region_points.append((orig_x, orig_y))
                print(f"Added point {len(self.region_points)}: ({orig_x}, {orig_y})")
    
    def handle_mouse_move(self, event):
        if not self.drawing_region or (not self.dragging_region and self.dragging_point is None):
            return
            
        # Get current mouse position
        move_x = event.position().x()
        move_y = event.position().y()
        
        # Convert to image coordinates
        img_x = move_x - self.pixmap_offset_x
        img_y = move_y - self.pixmap_offset_y
        
        # Check if move is within the scaled image bounds
        if (0 <= img_x < self.scaled_pixmap_size[0] and 
            0 <= img_y < self.scaled_pixmap_size[1]):
            
            # Scale to original image coordinates
            orig_x = int(img_x * self.scale_factor_x)
            orig_y = int(img_y * self.scale_factor_y)
            
            # Ensure coordinates are within bounds
            orig_x = max(0, min(orig_x, self.original_img_size[0] - 1))
            orig_y = max(0, min(orig_y, self.original_img_size[1] - 1))
            
            if self.dragging_point is not None and len(self.region_points) > self.dragging_point:
                # Move the dragged point
                self.region_points[self.dragging_point] = (orig_x, orig_y)
            elif self.dragging_region and len(self.region_points) == 4:
                # Move the entire region
                dx = orig_x - self.drag_start_pos[0]
                dy = orig_y - self.drag_start_pos[1]
                for i in range(4):
                    self.region_points[i] = (self.region_points[i][0] + dx, self.region_points[i][1] + dy)
                self.drag_start_pos = (orig_x, orig_y)
            
            # Update the display
            self.display_image(self.first_frame)
    
    def handle_mouse_release(self, event):
        self.dragging_region = False
        self.dragging_point = None
    
    def point_to_line_distance(self, point, line_start, line_end):
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        line_len = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_len == 0:
            return np.sqrt((x - x1)**2 + (y - y1)**2)
        u = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / (line_len ** 2)
        u = max(0, min(1, u))
        proj_x = x1 + u * (x2 - x1)
        proj_y = y1 + u * (y2 - y1)
        return np.sqrt((x - proj_x)**2 + (y - proj_y)**2)
    
    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    def start_counting(self):
        # Check if we have a valid region (either saved or newly drawn)
        if not self.has_valid_region and (not self.region_points_scaled or not self.region_fixed):
            self.show_colored_message(
                "Warning", 
                "Please draw and fix a counting region first", 
                QMessageBox.Icon.Warning
            )
            return
            
        self.is_counting = True
        # Make sure we're not frozen for region drawing
        self.freeze_for_region_drawing = False
        
        # Don't reset track_history, count_out, and counted_bags to continue from previous state
        if not hasattr(self, 'track_memory'):
            self.track_memory = {}
            
        self.update_count_labels()
        
        # Initialize video writer if not already writing
        if not self.video_writer:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"cement_bag_counting_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_filename, 
                fourcc, 
                15, 
                (self.FRAME_WIDTH, self.FRAME_HEIGHT)
            )
            self.video_path = output_filename
        
        self.start_counting_btn.setEnabled(False)
        self.stop_counting_btn.setEnabled(True)
        self.region_btn.setEnabled(False)
        
        # Force an immediate frame update to start processing
        self.update_frame()
    def stop_counting(self):
        self.is_counting = False
        # Don't freeze the frame, just stop processing
        
        # Don't release video writer to allow resuming later
        # Keep track_history, count_out, and counted_bags intact
        
        self.start_counting_btn.setEnabled(True)
        self.stop_counting_btn.setEnabled(False)
        self.region_btn.setEnabled(True)
        # QMessageBox.information(self, "Stopped", "Cement bag counting has stopped. Video is paused.")
        
    def reset_counting(self):
        """Reset the counting to zero"""
        self.count_out = 0
        self.counted_bags = set()
        self.track_history = defaultdict(lambda: [])
        self.track_memory = {}
        self.update_count_labels()
        
        # Also reset the video writer to start a new recording
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        # If counting is active, restart the video writer
        if self.is_counting:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"cement_bag_counting_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_filename, 
                fourcc, 
                15, 
                (self.FRAME_WIDTH, self.FRAME_HEIGHT)
            )
            self.video_path = output_filename
            
        self.show_colored_message(
            "Reset Complete",
            "Counting has been reset to zero.",
            QMessageBox.Icon.Information
        )
        
    def update_count_labels(self):
        self.count_out_label.setText(f"OUT: {self.count_out}")

        
    def detect_and_count(self, frame, model, region_points, track_history, frame_number=0):
        results = model.track(frame, persist=True, verbose=False, conf=0.5,iou=0.45)
        annotated_frame = frame.copy()

        # Track memory: {track_id: {'was_inside': bool, 'last_seen': int, 'last_y': int}}
        if not hasattr(self, 'track_memory'):
            self.track_memory = {}
        if not hasattr(self, 'counted_bags'):
            self.counted_bags = set()
        if not hasattr(self, 'subtracted_bags'):
            self.subtracted_bags = set()   # ✅ track IDs already subtracted

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            cement_indices = [i for i, cls_id in enumerate(class_ids) if cls_id == 0]

            for i in cement_indices:
                box = boxes[i]
                track_id = track_ids[i]

                x, y, w, h = box
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)

                # Skip small noise
                if w * h < 1500:
                    continue

                # Top-right corner of bbox
                top_right = (x2, y1)

                # Check if inside region
                is_inside = self.point_in_polygon(top_right, region_points)

                # Initialize track memory for new IDs
                if track_id not in self.track_memory:
                    self.track_memory[track_id] = {
                        'was_inside': False,
                        'last_seen': frame_number,
                        'last_y': y1
                    }

                # Get previous state
                was_inside = self.track_memory[track_id]['was_inside']
                last_y = self.track_memory[track_id]['last_y']

                # Movement direction
                dy = y1 - last_y   # positive = moving downward, negative = moving upward

                # ---------------------- COUNTING LOGIC ----------------------

                # Normal outside -> inside transition
                if not was_inside and is_inside:
                    if dy > 0:   # forward (top → bottom)
                        self.count_out += 1
                    elif dy < 0: # backward (bottom → top)
                        if track_id not in self.subtracted_bags:   # ✅ prevent multiple subtractions
                            self.count_out -= 1
                            self.subtracted_bags.add(track_id)

                    self.counted_bags.add(track_id)  # mark as counted
                    self.update_count_labels()

                # ✅ NEW LOGIC: If bag appears directly inside region (never outside before)
                elif is_inside and track_id not in self.counted_bags:
                    self.count_out += 1
                    self.counted_bags.add(track_id)
                    self.update_count_labels()

                # Inside -> outside (leaving region)
                elif was_inside and not is_inside:
                    if dy < 0:  # moving upward (bottom → top)
                        if track_id not in self.subtracted_bags:   # ✅ subtract only once
                            self.count_out -= 1
                            self.subtracted_bags.add(track_id)
                            self.update_count_labels()

                # ---------------------- UPDATE MEMORY ----------------------
                self.track_memory[track_id]['was_inside'] = is_inside
                self.track_memory[track_id]['last_seen'] = frame_number
                self.track_memory[track_id]['last_y'] = y1

                # Draw bounding box & info
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"ID:{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.circle(annotated_frame, top_right, 6, (255, 0, 0), -1)

        # Draw counting region
        for i in range(len(region_points)):
            start_point = region_points[i]
            end_point = region_points[(i + 1) % len(region_points)]
            cv2.line(annotated_frame, start_point, end_point, (0, 255, 0), 3)

        # Display count & timestamp
        cv2.putText(annotated_frame, f"Count OUT: {self.count_out}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, current_time, (annotated_frame.shape[1] - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Clean old IDs from memory
        MAX_TRACK_AGE = 50
        self.track_memory = {
            tid: mem for tid, mem in self.track_memory.items()
            if frame_number - mem['last_seen'] < MAX_TRACK_AGE
        }

        if self.video_writer:
            self.video_writer.write(annotated_frame)

        return annotated_frame



    def closeEvent(self, event):
        # Save region points if they exist
        if self.region_points and self.region_fixed:
            self.save_region_points()
            
        # Clean up resources
        if self.video_stream:
            self.video_stream.stop_stream()
        if self.video_writer:
            self.video_writer.release()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont()
    font.setFamily("Arial")
    font.setPointSize(10)
    app.setFont(font)
    
    window = CementBagOutCounterApp()
    window.show()
    sys.exit(app.exec())