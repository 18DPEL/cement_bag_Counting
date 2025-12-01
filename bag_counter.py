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
                             QFrame, QTabWidget)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPoint
from PyQt6.QtGui import QIcon, QImage, QPixmap, QKeyEvent, QPainter, QPen, QColor, QFont, QPalette

# Add this at the beginning to ensure the root path is in sys.path
def get_base_path():
    """Get the base path for both development and PyInstaller builds"""
    if getattr(sys, 'frozen', False):
        # Running in a bundle (PyInstaller)
        return sys._MEIPASS
    else:
        # Running in development
        return os.path.dirname(os.path.abspath(__file__))

# Add the base path to Python path
sys.path.insert(0, get_base_path())

def get_model_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = get_base_path()
    return os.path.join(base_path, relative_path)

class VideoStream(QObject):
    """Class to handle video streaming in a separate thread"""
    new_frame = pyqtSignal(np.ndarray, int)  # Added camera index
    stream_active = pyqtSignal(bool, int)    # Added camera index
    
    def __init__(self, rtsp_url, camera_index):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.camera_index = camera_index
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
            print(f"Error: Could not open video stream for camera {self.camera_index}")
            self.stream_active.emit(False, self.camera_index)
            return
        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        self.stream_active.emit(True, self.camera_index)
        
        # Get initial frame immediately
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            self.last_frame = frame
            self.new_frame.emit(frame, self.camera_index)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print(f"Frame grab failed for camera {self.camera_index}, retrying...")
                time.sleep(1)
                continue

            # Resize frame to fixed dimensions
            frame = cv2.resize(frame, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
            self.last_frame = frame
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame, timeout=1)
                self.new_frame.emit(frame, self.camera_index)
            except:
                pass

        cap.release()
        self.stream_active.emit(False, self.camera_index)
        
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
        try:
            self.setWindowIcon(QIcon(get_model_path('gui/icons/cropped-cropped-logo.ico')))
        except:
            pass  # Icon file not found, continue without icon
        self.setWindowTitle("Cement Bag Out Counter Application (Dual Camera)")
        self.setGeometry(100, 100, 1400, 800)
        
        # Configuration file path
        self.CONFIG_FILE = "cement_counter_config.json"
        self.REGION_FILE = "camera_regions.json"  # File for predefined regions
        
        # Full-screen display state
        self.fullscreen_camera = None  # None, 0, or 1 to indicate which camera is in fullscreen
        self.original_layout_state = None  # Store original layout state
        
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
            QTabWidget::pane {
                border: 1px solid #3498db;
                background-color: #34495e;
            }
            QTabBar::tab {
                background-color: #2c3e50;
                color: #ecf0f1;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3498db;
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
            .fullscreen-video {
                border: 3px solid #f39c12 !important;
            }
        """)
        
        # Frame size constants
        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480
        self.processing_width = self.FRAME_WIDTH
        self.processing_height = self.FRAME_HEIGHT
        
        # Model and tracking variables
        self.model = None
        self.track_history = [defaultdict(lambda: []), defaultdict(lambda: [])]  # Separate for each camera
        self.count_out = [0, 0]  # Separate counts for each camera
        self.counted_bags = [set(), set()]  # Track IDs of bags that have been counted for each camera
        self.region_points = [[], []]  # Points in original frame coordinates (4 points for quadrilateral)
        self.region_points_scaled = [[], []]  # Points in processing frame coordinates
        self.drawing_region = [False, False]
        self.region_fixed = [False, False]
        self.video_writer = [None, None]
        self.is_counting = [False, False]
        self.first_frame = [None, None]
        self.freeze_frame = [True, True]  # Start frozen
        self.dragging_region = [False, False]
        self.dragging_point = [None, None]
        self.line_thickness = 3
        self.point_radius = 8
        self.start_on_video_load = [False, False]
        self.video_path = [None, None]
        self.last_sent_count = [-1, -1]
        
        # Camera connection status
        self.camera_connected = [False, False]
        
        # Coordinate mapping variables
        self.original_img_size = [None, None]
        self.scaled_pixmap_size = [None, None]
        self.pixmap_offset_x = [0, 0]
        self.pixmap_offset_y = [0, 0]
        self.scale_factor_x = [1.0, 1.0]
        self.scale_factor_y = [1.0, 1.0]
        
        # Video streams
        self.video_streams = [None, None]
        self.current_frames = [None, None]
        
        # RTSP URLs for each camera
        self.rtsp_urls_camera1 = {
            "Camera 1 - Default": "rtsp://admin:Admin%4012345@192.168.53.127:554/cam/realmonitor?channel=1&subtype=0",
            "Camera 12 - Default": "rtsp://192.168.1.19:8554/mystream",
            "Camera 1 - Custom": ""
        }
        
        self.rtsp_urls_camera2 = {
            "Camera 2 - Default": "rtsp://admin:Admin%4012345@192.168.53.128:554/cam/realmonitor?channel=1&subtype=0",
            "Camera 22 - Default": "rtsp://192.168.1.19:8554/mystream",
            "Camera 2 - Custom": ""
        }
        
        # MQTT Client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        
        self.init_ui()
        self.start_mqtt()
        
        # Load saved region points if available
        # Try to load predefined regions from file
        self.load_region_points_from_file()
        self.load_region_points()
        
    def load_region_points_from_file(self, filename=None):
        """Load predefined region points from JSON file"""
        if filename is None:
            filename = self.REGION_FILE
            
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    regions = json.load(f)
                    # Handle both formats - the one with camera1/camera2 keys and the config format
                    if "camera1" in regions and len(regions["camera1"]) == 4:
                        # New format with camera1/camera2 keys
                        self.region_points[0] = [(pt[0], pt[1]) for pt in regions["camera1"]]
                        self.region_fixed[0] = True
                        if "camera2" in regions and len(regions["camera2"]) == 4:
                            self.region_points[1] = [(pt[0], pt[1]) for pt in regions["camera2"]]
                            self.region_fixed[1] = True
                    elif "region_points_0" in regions and len(regions["region_points_0"]) == 4:
                        # Config file format with region_points_0/1 keys
                        self.region_points[0] = [(pt[0], pt[1]) for pt in regions["region_points_0"]]
                        self.region_fixed[0] = True
                        if "region_points_1" in regions and len(regions["region_points_1"]) == 4:
                            self.region_points[1] = [(pt[0], pt[1]) for pt in regions["region_points_1"]]
                            self.region_fixed[1] = True
                print("Loaded predefined regions from file for both cameras")
                return True
        except Exception as e:
            print(f"Error loading region points from file: {e}")
        return False

    def load_region_points(self):
        """Load saved region points from config file if exists"""
        try:
            if os.path.exists(self.CONFIG_FILE):
                with open(self.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    loaded_any = False
                    for cam_idx in [0, 1]:
                        key = f'region_points_{cam_idx}'
                        if key in config:
                            self.region_points[cam_idx] = [(pt[0], pt[1]) for pt in config[key]]
                            self.region_fixed[cam_idx] = True
                            
                            # Calculate scaled coordinates
                            if self.original_img_size[cam_idx]:
                                scale_x = self.processing_width / self.original_img_size[cam_idx][0]
                                scale_y = self.processing_height / self.original_img_size[cam_idx][1]
                                self.region_points_scaled[cam_idx] = [
                                    (int(pt[0] * scale_x), int(pt[1] * scale_y)) 
                                    for pt in self.region_points[cam_idx]
                                ]
                                
                            print(f"Loaded saved region points for camera {cam_idx}:", self.region_points[cam_idx])
                            loaded_any = True
                    return loaded_any
        except Exception as e:
            print("Error loading region points:", e)
        return False    
        
    def save_region_points(self):
        """Save current region points to config file"""
        try:
            config = {
                'timestamp': datetime.datetime.now().isoformat()
            }
            for cam_idx in [0, 1]:
                config[f'region_points_{cam_idx}'] = self.region_points[cam_idx]
                
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            print("Saved region points to config file")
        except Exception as e:
            print("Error saving region points:", e)
    
    def init_ui(self):
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        
        # Left panel (controls)
        self.left_panel = QGroupBox("Controls")
        self.left_panel.setFixedWidth(250)
        left_layout = QVBoxLayout()
        
        # Camera configuration with tabs
        camera_config_group = QGroupBox("Camera Configuration")
        camera_config_group.setFixedHeight(280)
        camera_config_layout = QVBoxLayout()
        
        # Create tab widget for camera configuration
        self.camera_config_tabs = QTabWidget()
        
        # Camera 1 tab
        camera1_widget = QWidget()
        camera1_layout = QVBoxLayout()
        
        # Camera 1 status
        self.camera1_status = QLabel("Status: Disconnected")
        self.camera1_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
        camera1_layout.addWidget(self.camera1_status)
        
        # Camera 1 URL selection
        camera1_layout.addWidget(QLabel("Select Camera 1 URL:"))
        self.camera1_combo = QComboBox()
        self.camera1_combo.addItems(self.rtsp_urls_camera1.keys())
        camera1_layout.addWidget(self.camera1_combo)
        
        # Custom RTSP input for camera 1
        camera1_layout.addWidget(QLabel("Or enter custom RTSP URL:"))
        self.camera1_custom_input = QLineEdit()
        self.camera1_custom_input.setPlaceholderText("rtsp://username:password@ip:port/stream")
        camera1_layout.addWidget(self.camera1_custom_input)
        
        # Connect button for camera 1
        self.camera1_connect_btn = QPushButton("Connect Camera 1")
        self.camera1_connect_btn.clicked.connect(lambda: self.connect_camera(0))
        self.camera1_connect_btn.setStyleSheet("""
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
        camera1_layout.addWidget(self.camera1_connect_btn)
        
        camera1_widget.setLayout(camera1_layout)
        self.camera_config_tabs.addTab(camera1_widget, "Camera 1")
        
        # Camera 2 tab
        camera2_widget = QWidget()
        camera2_layout = QVBoxLayout()
        
        # Camera 2 status
        self.camera2_status = QLabel("Status: Disconnected")
        self.camera2_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
        camera2_layout.addWidget(self.camera2_status)
        
        # Camera 2 URL selection
        camera2_layout.addWidget(QLabel("Select Camera 2 URL:"))
        self.camera2_combo = QComboBox()
        self.camera2_combo.addItems(self.rtsp_urls_camera2.keys())
        camera2_layout.addWidget(self.camera2_combo)
        
        # Custom RTSP input for camera 2
        camera2_layout.addWidget(QLabel("Or enter custom RTSP URL:"))
        self.camera2_custom_input = QLineEdit()
        self.camera2_custom_input.setPlaceholderText("rtsp://username:password@ip:port/stream")
        camera2_layout.addWidget(self.camera2_custom_input)
        
        # Connect button for camera 2
        self.camera2_connect_btn = QPushButton("Connect Camera 2")
        self.camera2_connect_btn.clicked.connect(lambda: self.connect_camera(1))
        self.camera2_connect_btn.setStyleSheet("""
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
        camera2_layout.addWidget(self.camera2_connect_btn)
        
        camera2_widget.setLayout(camera2_layout)
        self.camera_config_tabs.addTab(camera2_widget, "Camera 2")
        
        camera_config_layout.addWidget(self.camera_config_tabs)
        camera_config_group.setLayout(camera_config_layout)
        
        # Region drawing controls with tabs
        region_group = QGroupBox("Counting Region Configuration")
        region_layout = QVBoxLayout()
        
        # Create tab widget for region drawing
        self.region_tabs = QTabWidget()
        
        # Camera 1 region tab
        region1_widget = QWidget()
        region1_layout = QVBoxLayout()
        
        self.region1_btn = QPushButton("Draw Region - Camera 1")
        self.region1_btn.clicked.connect(lambda: self.toggle_region_drawing(0))
        self.region1_btn.setEnabled(False)
        self.region1_btn.setStyleSheet("""
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
        region1_layout.addWidget(self.region1_btn)
        
        region1_widget.setLayout(region1_layout)
        self.region_tabs.addTab(region1_widget, "Camera 1 Region")
        
        # Camera 2 region tab
        region2_widget = QWidget()
        region2_layout = QVBoxLayout()
        
        self.region2_btn = QPushButton("Draw Region - Camera 2")
        self.region2_btn.clicked.connect(lambda: self.toggle_region_drawing(1))
        self.region2_btn.setEnabled(False)
        self.region2_btn.setStyleSheet("""
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
        region2_layout.addWidget(self.region2_btn)
        
        region2_widget.setLayout(region2_layout)
        self.region_tabs.addTab(region2_widget, "Camera 2 Region")
        
        region_layout.addWidget(self.region_tabs)
        region_group.setLayout(region_layout)
        
        # Counting controls
        count_group = QGroupBox("Bag Counting Controls")
        count_layout = QVBoxLayout()
        
        self.start_counting_btn = QPushButton("Start Counting (Both Cameras)")
        self.start_counting_btn.clicked.connect(self.start_all_counting)
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
        
        self.stop_counting_btn = QPushButton("Stop Counting (Both Cameras)")
        self.stop_counting_btn.clicked.connect(self.stop_all_counting)
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
        
        count_layout.addWidget(self.start_counting_btn)
        count_layout.addWidget(self.stop_counting_btn)
        count_group.setLayout(count_layout)
        
        # Count display
        count_display_group = QGroupBox("Bag Count")
        count_display_layout = QVBoxLayout()
        
        # Camera 1 count
        self.count_out_label1 = QLabel("Camera 1 OUT: 0")
        self.count_out_label1.setStyleSheet("font-size: 20px; font-weight: bold; border-radius: 2px;padding: 5px")
        self.count_out_label1.setObjectName("countDisplay")
        self.count_out_label1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        count_display_layout.addWidget(self.count_out_label1)
        
        # Camera 2 count
        self.count_out_label2 = QLabel("Camera 2 OUT: 0")
        self.count_out_label2.setStyleSheet("font-size: 20px; font-weight: bold; border-radius: 2px;padding: 5px")
        self.count_out_label2.setObjectName("countDisplay")
        self.count_out_label2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        count_display_layout.addWidget(self.count_out_label2)
        
        # Total count
        self.total_count_label = QLabel("Total OUT: 0")
        self.total_count_label.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #f39c12;
            background-color: #34495e;
            border: 3px solid #f39c12;
            border-radius: 5px;
            padding: 5px;
            text-align: center;
        """)
        self.total_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        count_display_layout.addWidget(self.total_count_label)
            
        count_display_group.setLayout(count_display_layout)
        
        # Add widgets to left layout
        left_layout.addWidget(camera_config_group)
        left_layout.addWidget(region_group)
        left_layout.addWidget(count_group)
        left_layout.addWidget(count_display_group)
        left_layout.addStretch()
        self.left_panel.setLayout(left_layout)
        
        # Right panel (video display)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create two video displays
        self.videos_layout = QHBoxLayout()
        self.video_labels = []
        self.video_frames = []
        
        for i in range(2):
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
            
            # Camera title
            camera_title = QLabel(f"Camera {i+1}")
            camera_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            camera_title.setStyleSheet("""
                font-weight: bold; 
                color: #3498db; 
                font-size: 14px;
                margin: 0;
                padding: 0;
            """)
            camera_title.setFixedHeight(25)
            video_layout.addWidget(camera_title)
            
            # Video display
            video_label = QLabel()
            video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            video_label.setStyleSheet("background-color: black;")
            video_label.setMinimumSize(640, 480)
            
            # Set placeholder text
            video_label.setText(f"Camera {i+1}\nNot Connected")
            video_label.setStyleSheet("""
                background-color: black; 
                color: white; 
                font-size: 16px; 
                font-weight: bold;
            """)
            
            # Add click functionality for fullscreen toggle
            video_label.mousePressEvent = lambda event, cam_idx=i: self.toggle_fullscreen_camera(cam_idx)
            
            video_layout.addWidget(video_label)
            
            self.videos_layout.addWidget(video_frame)
            self.video_labels.append(video_label)
            self.video_frames.append(video_frame)
            
        self.right_layout.addLayout(self.videos_layout)
        self.right_panel.setLayout(self.right_layout)
        
        # Add panels to main layout
        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.right_panel)
        
        # Timer for video display
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
        # Load model
        self.load_model()
        
    def toggle_fullscreen_camera(self, cam_idx):
        """Toggle fullscreen display for specific camera"""
        # Don't allow fullscreen if camera is not connected
        if not self.camera_connected[cam_idx]:
            return
            
        if self.fullscreen_camera == cam_idx:
            # Return to normal view
            self.exit_fullscreen()
        elif self.fullscreen_camera is None:
            # Enter fullscreen for this camera
            self.enter_fullscreen(cam_idx)
        else:
            # Switch from one camera fullscreen to another
            self.exit_fullscreen()
            self.enter_fullscreen(cam_idx)
    
    def enter_fullscreen(self, cam_idx):
        """Enter fullscreen mode for specific camera"""
        self.fullscreen_camera = cam_idx
        
        # Hide the left panel
        self.left_panel.hide()
        
        # Hide the other camera
        other_cam_idx = 1 - cam_idx
        self.video_frames[other_cam_idx].hide()
        
        # Update the fullscreen camera frame styling
        self.video_frames[cam_idx].setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border: 3px solid #f39c12;
                border-radius: 5px;
            }
        """)
        
        # Resize the video label to take full space
        self.video_labels[cam_idx].setMinimumSize(1200, 800)
        
        # Add fullscreen indicator
        title_label = self.video_frames[cam_idx].findChild(QLabel)
        if title_label:
            title_label.setText(f"Camera {cam_idx+1} - FULLSCREEN (Click to return)")
            title_label.setStyleSheet("font-weight: bold; color: #f39c12; font-size: 16px;")
        
        print(f"Entered fullscreen mode for Camera {cam_idx+1}")
    
    def exit_fullscreen(self):
        """Exit fullscreen mode and return to dual camera view"""
        if self.fullscreen_camera is None:
            return
            
        cam_idx = self.fullscreen_camera
        self.fullscreen_camera = None
        
        # Show the left panel
        self.left_panel.show()
        
        # Show both camera frames
        for i in range(2):
            self.video_frames[i].show()
        
        # Reset styling for both frames
        for i in range(2):
            self.video_frames[i].setStyleSheet("""
                QFrame {
                    background-color: #2c3e50;
                    border: 3px solid #3498db;
                    border-radius: 5px;
                }
            """)
        
        # Reset video label sizes
        for i in range(2):
            self.video_labels[i].setMinimumSize(640, 480)
        
        # Reset camera titles
        for i in range(2):
            title_label = self.video_frames[i].findChild(QLabel)
            if title_label:
                title_label.setText(f"Camera {i+1}")
                title_label.setStyleSheet("font-weight: bold; color: #3498db; font-size: 14px;")
        
        print("Exited fullscreen mode")
    
    def connect_camera(self, cam_idx):
        """Connect to RTSP camera"""
        # Get RTSP URL based on camera index
        if cam_idx == 0:
            if self.camera1_custom_input.text():
                rtsp_url = self.camera1_custom_input.text()
            else:
                selected_camera = self.camera1_combo.currentText()
                rtsp_url = self.rtsp_urls_camera1[selected_camera]
            connect_btn = self.camera1_connect_btn
            status_label = self.camera1_status
        else:
            if self.camera2_custom_input.text():
                rtsp_url = self.camera2_custom_input.text()
            else:
                selected_camera = self.camera2_combo.currentText()
                rtsp_url = self.rtsp_urls_camera2[selected_camera]
            connect_btn = self.camera2_connect_btn
            status_label = self.camera2_status
            
        if not rtsp_url or rtsp_url == "":
            QMessageBox.warning(self, "Warning", f"Please enter a valid RTSP URL for Camera {cam_idx+1}")
            return
            
        # Update UI to show connecting
        connect_btn.setEnabled(False)
        connect_btn.setText(f"Connecting...")
        status_label.setText("Status: Connecting...")
        status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        
        # Show connecting message on video display
        self.display_connecting_message(cam_idx)
        
        # Stop any existing stream for this camera
        if self.video_streams[cam_idx]:
            self.video_streams[cam_idx].stop_stream()
            
        # Start new stream
        self.video_streams[cam_idx] = VideoStream(rtsp_url, cam_idx)
        self.video_streams[cam_idx].new_frame.connect(self.handle_new_frame)
        self.video_streams[cam_idx].stream_active.connect(self.handle_stream_status)
        self.video_streams[cam_idx].start_stream()
        
        # Start timer for frame updates if not already running
        if not self.timer.isActive():
            self.timer.start(30)
        
        # Reset states for this camera
        self.reset_region_state(cam_idx)
        
    def reset_region_state(self, cam_idx):
        """Reset all region-related states for specific camera"""
        # Only reset if we don't have predefined regions
        if not self.region_points[cam_idx] or len(self.region_points[cam_idx]) != 4:
            self.region_points[cam_idx] = []
            self.region_points_scaled[cam_idx] = []
            self.region_fixed[cam_idx] = False
            
        self.drawing_region[cam_idx] = False
        self.dragging_region[cam_idx] = False
        self.dragging_point[cam_idx] = None
        self.freeze_frame[cam_idx] = True  # Freeze on first frame
        
        # Update region button
        if cam_idx == 0:
            self.region1_btn.setEnabled(True if self.camera_connected[0] else False)
            if self.region_fixed[0]:
                self.region1_btn.setText("Adjust Region - Camera 1")
                self.region1_btn.setStyleSheet("""
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
            else:
                self.region1_btn.setText("Draw Region - Camera 1")
                self.region1_btn.setStyleSheet("""
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
        else:
            self.region2_btn.setEnabled(True if self.camera_connected[1] else False)
            if self.region_fixed[1]:
                self.region2_btn.setText("Adjust Region - Camera 2")
                self.region2_btn.setStyleSheet("""
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
            else:
                self.region2_btn.setText("Draw Region - Camera 2")
                self.region2_btn.setStyleSheet("""
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
        
        # Setup mouse handlers
        self.setup_camera_click_handler(cam_idx)
        
        # Update start button availability
        self.update_start_button_state()
        def auto_set_regions(self):
            """Automatically set regions for both cameras if available"""
            if not all(self.camera_connected):
                return
                
            if not all(self.first_frame):
                return
                
            # Check if we have predefined regions for both cameras
            if not (self.region_points[0] and self.region_points[1]):
                return
                
            for cam_idx in [0, 1]:
                if self.region_points[cam_idx] and self.region_fixed[cam_idx]:
                    # Calculate scaled coordinates
                    if self.original_img_size[cam_idx]:
                        scale_x = self.processing_width / self.original_img_size[cam_idx][0]
                        scale_y = self.processing_height / self.original_img_size[cam_idx][1]
                        self.region_points_scaled[cam_idx] = [
                            (int(pt[0] * scale_x), int(pt[1] * scale_y)) 
                            for pt in self.region_points[cam_idx]
                        ]
                    
                    # Display the frame with region
                    display_frame = self.first_frame[cam_idx].copy()
                    for i in range(len(self.region_points[cam_idx])):
                        start_point = self.region_points[cam_idx][i]
                        end_point = self.region_points[cam_idx][(i + 1) % len(self.region_points[cam_idx])]
                        cv2.line(display_frame, start_point, end_point, (0, 255, 0), self.line_thickness)
                        cv2.circle(display_frame, start_point, self.point_radius, (0, 255, 0), -1)
                    self.display_image(display_frame, cam_idx)
                    
                    # Update UI
                    if cam_idx == 0:
                        self.region1_btn.setText("Adjust Region - Camera 1")
                        self.region1_btn.setStyleSheet("""
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
                    else:
                        self.region2_btn.setText("Adjust Region - Camera 2")
                        self.region2_btn.setStyleSheet("""
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
            
            self.update_start_button_state()
            self.show_colored_message(
                "Success",
                "Counting regions automatically loaded from file for both cameras!",
                QMessageBox.Icon.Information
            )

    def handle_stream_status(self, active, cam_idx):
        """Handle stream connection status"""
        if active:
            # Connection successful
            self.camera_connected[cam_idx] = True
            
            if cam_idx == 0:
                self.camera1_connect_btn.setEnabled(True)
                self.camera1_connect_btn.setText("Reconnect Camera 1")
                self.camera1_status.setText("Status: Connected")
                self.camera1_status.setStyleSheet("color: #27ae60; font-weight: bold;")
                self.region1_btn.setEnabled(True)
            else:
                self.camera2_connect_btn.setEnabled(True)
                self.camera2_connect_btn.setText("Reconnect Camera 2")
                self.camera2_status.setText("Status: Connected")
                self.camera2_status.setStyleSheet("color: #27ae60; font-weight: bold;")
                self.region2_btn.setEnabled(True)
            
            # Setup click handler for fullscreen and region drawing
            self.setup_camera_click_handler(cam_idx)
            
            # Get the first frame and store it
            if self.video_streams[cam_idx]:
                self.first_frame[cam_idx] = self.video_streams[cam_idx].get_last_frame()
                if self.first_frame[cam_idx] is not None:
                    # Display the first frame frozen
                    self.display_image(self.first_frame[cam_idx], cam_idx)
                    
                    # If we have predefined points, automatically fix them
                    if self.region_points[cam_idx] and len(self.region_points[cam_idx]) == 4:
                        self.region_fixed[cam_idx] = True
                        self.drawing_region[cam_idx] = False
                        
                        # Calculate scaled coordinates for processing frame
                        if self.original_img_size[cam_idx]:
                            scale_x = self.processing_width / self.original_img_size[cam_idx][0]
                            scale_y = self.processing_height / self.original_img_size[cam_idx][1]
                            
                            self.region_points_scaled[cam_idx] = [
                                (int(pt[0] * scale_x), int(pt[1] * scale_y)) 
                                for pt in self.region_points[cam_idx]
                            ]
                        
                        # Update UI
                        if cam_idx == 0:
                            self.region1_btn.setText("Adjust Region - Camera 1")
                            self.region1_btn.setStyleSheet("""
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
                        else:
                            self.region2_btn.setText("Adjust Region - Camera 2")
                            self.region2_btn.setStyleSheet("""
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
                        
                        # Display the frame with region
                        display_frame = self.first_frame[cam_idx].copy()
                        for i in range(len(self.region_points[cam_idx])):
                            start_point = self.region_points[cam_idx][i]
                            end_point = self.region_points[cam_idx][(i + 1) % len(self.region_points[cam_idx])]
                            cv2.line(display_frame, start_point, end_point, (0, 255, 0), self.line_thickness)
                            cv2.circle(display_frame, start_point, self.point_radius, (0, 255, 0), -1)
                        self.display_image(display_frame, cam_idx)
                        
                        self.show_colored_message(
                            "Success",
                            f"Camera {cam_idx+1} connected successfully! Predefined counting region automatically loaded.\n"
                            "You can adjust the region or start counting.\n"
                            "Click on camera display to toggle fullscreen view.",
                            QMessageBox.Icon.Information
                        )
                    else:
                        self.show_colored_message(
                            "Success",
                            f"Camera {cam_idx+1} connected successfully! Click 'Draw Region' to set the counting area.\n"
                            "Click on camera display to toggle fullscreen view.",
                            QMessageBox.Icon.Information
                        )
                    
                    # Try to auto-set regions if both cameras are connected
                    if all(self.camera_connected) and all(self.first_frame):
                        QTimer.singleShot(1000, self.auto_set_regions)        



    def handle_new_frame(self, frame, cam_idx):
        """Handle new frame from video stream"""
        if self.freeze_frame[cam_idx] and self.first_frame[cam_idx] is None:
            # Store the first frame and freeze
            self.first_frame[cam_idx] = frame.copy()
        self.current_frames[cam_idx] = frame
        
    def update_frame(self):
        """Update video display for both cameras"""
        for cam_idx in [0, 1]:
            # Skip hidden cameras in fullscreen mode
            if self.fullscreen_camera is not None and self.fullscreen_camera != cam_idx:
                continue
                
            if not self.camera_connected[cam_idx] or self.current_frames[cam_idx] is None:
                continue
                
            # If we're frozen on first frame (before region is fixed or during drawing)
            if self.freeze_frame[cam_idx] and self.first_frame[cam_idx] is not None:
                display_frame = self.first_frame[cam_idx].copy()
                
                # Draw the region if we have points
                if len(self.region_points[cam_idx]) > 1:
                    # Draw lines between points
                    for i in range(len(self.region_points[cam_idx])):
                        start_point = self.region_points[cam_idx][i]
                        end_point = self.region_points[cam_idx][(i + 1) % len(self.region_points[cam_idx])]
                        cv2.line(display_frame, start_point, end_point, 
                                 (0, 255, 0) if self.region_fixed[cam_idx] else (0, 255, 255), 
                                 self.line_thickness)
                    
                    # Draw endpoints
                    for i, pt in enumerate(self.region_points[cam_idx]):
                        color = (0, 255, 0) if self.region_fixed[cam_idx] else (0, 255, 255)
                        cv2.circle(display_frame, pt, self.point_radius, color, -1)
                        cv2.putText(display_frame, str(i+1), (pt[0]+10, pt[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                self.display_image(display_frame, cam_idx)
                continue
                
            # If counting is active, process the frame
            if self.is_counting[cam_idx] and self.region_points_scaled[cam_idx] and self.region_fixed[cam_idx]:
                # Ensure frame is the correct size
                processing_frame = cv2.resize(self.current_frames[cam_idx], (self.FRAME_WIDTH, self.FRAME_HEIGHT))
                
                annotated_frame = self.detect_and_count(
                    processing_frame, self.model, self.region_points_scaled[cam_idx], 
                    self.track_history[cam_idx], cam_idx
                )
                
                # Display the annotated frame
                if annotated_frame is not None:
                    self.display_image(annotated_frame, cam_idx)
                
                # Periodically upload count to server
                if time.time() % 15 < 0.1:  # About every 15 seconds
                    self.upload_count_to_server()
            else:
                # Just show the current frame with confirmed region if available
                display_frame = self.current_frames[cam_idx].copy()
                if len(self.region_points[cam_idx]) > 1:
                    color = (0, 255, 0) if self.region_fixed[cam_idx] else (0, 255, 255)
                    for i in range(len(self.region_points[cam_idx])):
                        start_point = self.region_points[cam_idx][i]
                        end_point = self.region_points[cam_idx][(i + 1) % len(self.region_points[cam_idx])]
                        cv2.line(display_frame, start_point, end_point, color, self.line_thickness)
                    for pt in self.region_points[cam_idx]:
                        cv2.circle(display_frame, pt, self.point_radius, color, -1)
                self.display_image(display_frame, cam_idx)
    
    def display_image(self, img, cam_idx):
        """Display image on video label"""
        if img is None:
            return
            
        # Ensure image is the correct size
        img = cv2.resize(img, (self.FRAME_WIDTH, self.FRAME_HEIGHT))
        
        # Convert the image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original image size for coordinate mapping
        self.original_img_size[cam_idx] = (self.FRAME_WIDTH, self.FRAME_HEIGHT)
        
        # Convert to QImage
        bytes_per_line = 3 * self.FRAME_WIDTH
        q_img = QImage(img_rgb.data, self.FRAME_WIDTH, self.FRAME_HEIGHT, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Create pixmap
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale the pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_labels[cam_idx].size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        
        # Store scaling information
        self.scaled_pixmap_size[cam_idx] = (scaled_pixmap.width(), scaled_pixmap.height())
        self.scale_factor_x[cam_idx] = self.FRAME_WIDTH / scaled_pixmap.width() if scaled_pixmap.width() > 0 else 1.0
        self.scale_factor_y[cam_idx] = self.FRAME_HEIGHT / scaled_pixmap.height() if scaled_pixmap.height() > 0 else 1.0
        
        # Calculate offsets for centered display
        self.pixmap_offset_x[cam_idx] = (self.video_labels[cam_idx].width() - scaled_pixmap.width()) // 2
        self.pixmap_offset_y[cam_idx] = (self.video_labels[cam_idx].height() - scaled_pixmap.height()) // 2
        
        self.video_labels[cam_idx].setPixmap(scaled_pixmap)
    
    def toggle_region_drawing(self, cam_idx):
        """Toggle region drawing for specific camera"""
        if self.first_frame[cam_idx] is None:
            # Try to get a new frame if none is available
            if self.video_streams[cam_idx]:
                self.first_frame[cam_idx] = self.video_streams[cam_idx].get_last_frame()
                if self.first_frame[cam_idx] is None:
                    QMessageBox.warning(self, "Warning", f"No frame available to draw on for Camera {cam_idx+1}")
                    return
            else:
                QMessageBox.warning(self, "Warning", f"Camera {cam_idx+1} is not connected")
                return
                
        region_btn = self.region1_btn if cam_idx == 0 else self.region2_btn

        if not self.drawing_region[cam_idx]:
            # Start drawing mode
            self.drawing_region[cam_idx] = True
            self.region_fixed[cam_idx] = False
            self.region_points[cam_idx] = []
            self.region_points_scaled[cam_idx] = []
            self.freeze_frame[cam_idx] = True  # Ensure we're frozen on this frame
            
            # Update UI
            region_btn.setText(f"Fix Region - Camera {cam_idx+1}")
            region_btn.setStyleSheet("""
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
            
            # Setup unified mouse handler
            self.setup_camera_click_handler(cam_idx)
            
            # Display the frozen frame
            self.display_image(self.first_frame[cam_idx], cam_idx)
            
            self.show_colored_message(
                "Instructions",
                f"Click four points to draw the counting region for Camera {cam_idx+1}.\n"
                "Click points in order (e.g., clockwise or counter-clockwise).\n"
                "Then press 'Fix Region' to confirm.\n"
                "You can drag points to adjust the region.\n"
                "Fullscreen toggle is disabled during region drawing."
            )
        else:
            # Check if we have a complete region (4 points)
            if len(self.region_points[cam_idx]) != 4:
                self.show_colored_message("Warning", f"Please draw a complete region (four points) first for Camera {cam_idx+1}", QMessageBox.Icon.Warning)
                return
                
            # Toggle fixed state
            self.region_fixed[cam_idx] = not self.region_fixed[cam_idx]
            
            if self.region_fixed[cam_idx]:
                # Calculate scaled coordinates for processing frame
                if self.original_img_size[cam_idx]:
                    scale_x = self.processing_width / self.original_img_size[cam_idx][0]
                    scale_y = self.processing_height / self.original_img_size[cam_idx][1]
                    
                    self.region_points_scaled[cam_idx] = [
                        (int(pt[0] * scale_x), int(pt[1] * scale_y)) 
                        for pt in self.region_points[cam_idx]
                    ]
                
                # Save the region points
                self.save_region_points()
                
                # End drawing mode
                self.drawing_region[cam_idx] = False
                
                # Update button
                region_btn.setText(f"Adjust Region - Camera {cam_idx+1}")
                region_btn.setStyleSheet("""
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
                
                print(f"Region fixed for Camera {cam_idx+1} with points:")
                for i, pt in enumerate(self.region_points[cam_idx]):
                    print(f"  Point {i+1}: {pt}")
                print(f"Scaled points:")
                for i, pt in enumerate(self.region_points_scaled[cam_idx]):
                    print(f"  Point {i+1}: {pt}")
                
                self.show_colored_message(
                    "Success",
                    f"Counting region fixed for Camera {cam_idx+1}!\n"
                    "You can now start counting when ready.\n"
                    "Click on camera display to toggle fullscreen view.",
                    QMessageBox.Icon.Information
                )
            else:
                # Region is now adjustable (back to drawing mode)
                self.drawing_region[cam_idx] = True
                region_btn.setText(f"Fix Region - Camera {cam_idx+1}")
                region_btn.setStyleSheet("""
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
            
            # Update mouse handlers based on new state
            self.setup_camera_click_handler(cam_idx)
        
        # Update start button state
        self.update_start_button_state() 
    
    def setup_camera_click_handler(self, cam_idx):
        """Setup a unified mouse handler that handles both fullscreen and region drawing"""
        def mouse_handler(event):
            # Check if we're in region drawing mode
            if self.drawing_region[cam_idx]:
                self.handle_mouse_click(event, cam_idx)
            else:
                # Only toggle fullscreen if not in region drawing mode
                self.toggle_fullscreen_camera(cam_idx)
        
        # Assign the unified handler
        self.video_labels[cam_idx].mousePressEvent = mouse_handler
        
        # Also set up move and release handlers when in drawing mode
        if self.drawing_region[cam_idx]:
            self.video_labels[cam_idx].mouseMoveEvent = lambda e: self.handle_mouse_move(e, cam_idx)
            self.video_labels[cam_idx].mouseReleaseEvent = lambda e: self.handle_mouse_release(e, cam_idx)
        else:
            # Clear move and release handlers when not in drawing mode
            self.video_labels[cam_idx].mouseMoveEvent = None
            self.video_labels[cam_idx].mouseReleaseEvent = None
        
    def update_start_button_state(self):
        """Update start button state based on camera connections and regions"""
        both_connected = self.camera_connected[0] and self.camera_connected[1]
        both_regions_fixed = self.region_fixed[0] and self.region_fixed[1]
        any_connected = self.camera_connected[0] or self.camera_connected[1]
        any_region_fixed = self.region_fixed[0] or self.region_fixed[1]
        
        # Enable start button if at least one camera is connected and has a region
        can_start = any_connected and any_region_fixed
        
        if both_connected and both_regions_fixed:
            self.start_counting_btn.setText("Start Counting (Both Cameras)")
        elif can_start:
            active_cameras = []
            if self.camera_connected[0] and self.region_fixed[0]:
                active_cameras.append("Camera 1")
            if self.camera_connected[1] and self.region_fixed[1]:
                active_cameras.append("Camera 2")
            self.start_counting_btn.setText(f"Start Counting ({', '.join(active_cameras)})")
        else:
            self.start_counting_btn.setText("Start Counting (Both Cameras)")
            
        self.start_counting_btn.setEnabled(can_start)
        
    def display_connecting_message(self, cam_idx):
        """Display connecting message on video label"""
        # Create a black image with connecting message using fixed size
        img = np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(img, f"Connecting to", 
                   (200, self.FRAME_HEIGHT//2 - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(img, f"Camera {cam_idx+1}...", 
                   (180, self.FRAME_HEIGHT//2 + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        self.display_image(img, cam_idx)
        
    def handle_mouse_click(self, event, cam_idx):
        """Handle mouse click for region drawing"""
        if not self.drawing_region[cam_idx]:
            return
            
        # Get click position relative to the video label
        click_x = event.position().x()
        click_y = event.position().y()
        
        # Convert to image coordinates
        img_x = click_x - self.pixmap_offset_x[cam_idx]
        img_y = click_y - self.pixmap_offset_y[cam_idx]
        
        # Check if click is within the scaled image bounds
        if (0 <= img_x < self.scaled_pixmap_size[cam_idx][0] and 
            0 <= img_y < self.scaled_pixmap_size[cam_idx][1]):
            
            # Scale to original image coordinates
            orig_x = int(img_x * self.scale_factor_x[cam_idx])
            orig_y = int(img_y * self.scale_factor_y[cam_idx])
            
            # Ensure coordinates are within bounds
            orig_x = max(0, min(orig_x, self.original_img_size[cam_idx][0] - 1))
            orig_y = max(0, min(orig_y, self.original_img_size[cam_idx][1] - 1))
            
            # Check if we're clicking near an existing point
            if len(self.region_points[cam_idx]) > 0:
                for i, pt in enumerate(self.region_points[cam_idx]):
                    distance = np.sqrt((orig_x - pt[0])**2 + (orig_y - pt[1])**2)
                    if distance < 20:  # Clicked near a point
                        self.dragging_point[cam_idx] = i
                        return
                
                # Check if we're clicking near a line (to move entire region)
                if len(self.region_points[cam_idx]) == 4:
                    for i in range(4):
                        line_start = self.region_points[cam_idx][i]
                        line_end = self.region_points[cam_idx][(i + 1) % 4]
                        if self.point_to_line_distance((orig_x, orig_y), line_start, line_end) < 10:
                            self.dragging_region[cam_idx] = True
                            self.drag_start_pos = (orig_x, orig_y)
                            return
            
            # If we're not dragging, add new points (up to 4)
            if len(self.region_points[cam_idx]) < 4:
                self.region_points[cam_idx].append((orig_x, orig_y))
                print(f"Added point {len(self.region_points[cam_idx])} for Camera {cam_idx+1}: ({orig_x}, {orig_y})")
    
    def handle_mouse_move(self, event, cam_idx):
        """Handle mouse move for region drawing"""
        if not self.drawing_region[cam_idx] or (not self.dragging_region[cam_idx] and self.dragging_point[cam_idx] is None):
            return
            
        # Get current mouse position
        move_x = event.position().x()
        move_y = event.position().y()
        
        # Convert to image coordinates
        img_x = move_x - self.pixmap_offset_x[cam_idx]
        img_y = move_y - self.pixmap_offset_y[cam_idx]
        
        # Check if move is within the scaled image bounds
        if (0 <= img_x < self.scaled_pixmap_size[cam_idx][0] and 
            0 <= img_y < self.scaled_pixmap_size[cam_idx][1]):
            
            # Scale to original image coordinates
            orig_x = int(img_x * self.scale_factor_x[cam_idx])
            orig_y = int(img_y * self.scale_factor_y[cam_idx])
            
            # Ensure coordinates are within bounds
            orig_x = max(0, min(orig_x, self.original_img_size[cam_idx][0] - 1))
            orig_y = max(0, min(orig_y, self.original_img_size[cam_idx][1] - 1))
            
            if self.dragging_point[cam_idx] is not None and len(self.region_points[cam_idx]) > self.dragging_point[cam_idx]:
                # Move the dragged point
                self.region_points[cam_idx][self.dragging_point[cam_idx]] = (orig_x, orig_y)
            elif self.dragging_region[cam_idx] and len(self.region_points[cam_idx]) == 4:
                # Move the entire region
                dx = orig_x - self.drag_start_pos[0]
                dy = orig_y - self.drag_start_pos[1]
                for i in range(4):
                    self.region_points[cam_idx][i] = (self.region_points[cam_idx][i][0] + dx, self.region_points[cam_idx][i][1] + dy)
                self.drag_start_pos = (orig_x, orig_y)
            
            # Update the display
            self.display_image(self.first_frame[cam_idx], cam_idx)
    
    def handle_mouse_release(self, event, cam_idx):
        """Handle mouse release for region drawing"""
        self.dragging_region[cam_idx] = False
        self.dragging_point[cam_idx] = None
    
    def point_to_line_distance(self, point, line_start, line_end):
        """Calculate distance from point to line"""
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
        
    def start_all_counting(self):
        """Start counting for all connected cameras with regions"""
        print("Attempting to start counting for all cameras")
        # Exit fullscreen mode if active
        if self.fullscreen_camera is not None:
            self.exit_fullscreen()
            
        started_cameras = []
        
        for cam_idx in [0, 1]:
            if self.camera_connected[cam_idx] and self.region_fixed[cam_idx]:
                self.start_counting(cam_idx)
                started_cameras.append(f"Camera {cam_idx+1}")
                
        if started_cameras:
            print(f"Successfully started counting on: {', '.join(started_cameras)}")
            self.show_colored_message(
                "Counting Started",
                f"Started counting on: {', '.join(started_cameras)}",
                QMessageBox.Icon.Information
            )
        else:
            print("No cameras available for counting")
            self.show_colored_message(
                "Warning",
                "No cameras available for counting. Please connect cameras and draw regions first.",
                QMessageBox.Icon.Warning
            )
            
    def stop_all_counting(self):
        """Stop counting for all cameras"""
        print("Attempting to stop counting for all cameras")
        stopped_cameras = []
        
        for cam_idx in [0, 1]:
            if self.is_counting[cam_idx]:
                self.stop_counting(cam_idx)
                stopped_cameras.append(f"Camera {cam_idx+1}")
                
        if stopped_cameras:
            print(f"Successfully stopped counting on: {', '.join(stopped_cameras)}")
            self.show_colored_message(
                "Counting Stopped",
                f"Stopped counting on: {', '.join(stopped_cameras)}",
                QMessageBox.Icon.Information
            )
            
    def start_counting(self, cam_idx):
        """Start counting for specific camera"""
        print(f"Attempting to start counting for camera {cam_idx}")
        print(f"Region points scaled: {self.region_points_scaled[cam_idx]}")
        print(f"Region fixed: {self.region_fixed[cam_idx]}")
        
        if not self.region_points_scaled[cam_idx] or not self.region_fixed[cam_idx]:
            print("Cannot start - region not properly configured")
            return
            
        self.is_counting[cam_idx] = True
        self.freeze_frame[cam_idx] = False  # Unfreeze to show live stream
        self.track_history[cam_idx] = defaultdict(lambda: [])
        self.count_out[cam_idx] = 0
        self.counted_bags[cam_idx] = set()
        self.update_count_labels()
        
        # Initialize video writer
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"cement_bag_counting_cam{cam_idx+1}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer[cam_idx] = cv2.VideoWriter(
            output_filename, 
            fourcc, 
            15, 
            (self.FRAME_WIDTH, self.FRAME_HEIGHT)
        )
        if not self.video_writer[cam_idx].isOpened():
            print(f"Error: Could not open video writer for camera {cam_idx}")
        else:
            print(f"Video writer initialized for camera {cam_idx}")
        
        self.video_path[cam_idx] = output_filename
        
        # Update UI
        self.stop_counting_btn.setEnabled(True)
        if cam_idx == 0:
            self.region1_btn.setEnabled(False)
        else:
            self.region2_btn.setEnabled(False)
            
        # Check if both cameras are counting
        if all(self.is_counting):
            self.start_counting_btn.setEnabled(False)
        
        print(f"Counting successfully started for camera {cam_idx}")
        
    def stop_counting(self, cam_idx):
        """Stop counting for specific camera"""
        print(f"Stopping counting for camera {cam_idx}")
        self.is_counting[cam_idx] = False
        self.freeze_frame[cam_idx] = True  # Freeze frame when stopped
        if self.video_writer[cam_idx]:
            self.video_writer[cam_idx].release()
            self.video_writer[cam_idx] = None
            
        # Update UI
        if cam_idx == 0:
            self.region1_btn.setEnabled(True)
        else:
            self.region2_btn.setEnabled(True)
            
        # Check if no cameras are counting
        if not any(self.is_counting):
            self.stop_counting_btn.setEnabled(False)
            self.update_start_button_state()
        
        print(f"Counting successfully stopped for camera {cam_idx}")
    
    def update_count_labels(self):
        """Update count display labels"""
        self.count_out_label1.setText(f"Camera 1 OUT: {self.count_out[0]}")
        self.count_out_label2.setText(f"Camera 2 OUT: {self.count_out[1]}")
        total = sum(self.count_out)
        self.total_count_label.setText(f"Total OUT: {total}")
        
    def show_colored_message(self, title, text, icon=QMessageBox.Icon.Information):
        """Show colored message box"""
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

                if not any(self.camera_connected):
                    print("⚠️ No cameras connected. Will auto-start after connection.")
                    self.start_on_video_load = [True, True]
                    return

                for cam_idx in [0, 1]:
                    self.count_out[cam_idx] = 0
                    self.counted_bags[cam_idx].clear()
                    self.track_history[cam_idx].clear()
                    
                QTimer.singleShot(0, self.start_all_counting)

            elif action == "stop":
                print("🛑 MQTT Action: Stop")
                # Get wagon_no and loading_dock from the MQTT message
                wagon_no = data.get("wagon_no", "unknown")
                loading_dock = data.get("loading_dock", "WL 2")  # Default to "WL 2" if not provided
                
                # Send final count with additional information
                self.send_stop_payload(wagon_no, loading_dock)
                
                QTimer.singleShot(0, self.stop_all_counting)

        except Exception as e:
            print("❌ MQTT message error:", e)

    def send_stop_payload(self, wagon_no, loading_dock):
        try:
            total_count = sum(self.count_out)
            payload = {
                "count": total_count,
                "loading_dock": loading_dock,
                "wagon_no": wagon_no,
                "video": "combined_output.mp4",  # You might want to combine videos
                "status": "completed"  # Indicate this is a final count
            }
            print("Sending payload:", payload)
            response = requests.post("https://shipeasy.tech/cement/public/api/get_load", 
                                   data=payload,
                                   timeout=1)
            print("Final count sent to server. Response:", response.json())
            self.last_sent_count = self.count_out.copy()
        except Exception as e:
            print("Error sending final count:", e)

    def upload_count_to_server(self):
        if any(count != last_sent for count, last_sent in zip(self.count_out, self.last_sent_count)):
            try:
                total_count = sum(self.count_out)
                payload = {
                    "count": total_count,
                    "loading_dock": "WL 2",
                    "video": "combined_output.mp4",
                    "status": "in_progress"  # Indicate this is an intermediate count
                }
                response = requests.post("https://shipeasy.tech/cement/public/api/get_load", 
                                       data=payload,
                                       timeout=2)
                print("Server response:", response.json())
                self.last_sent_count = self.count_out.copy()
            except Exception as e:
                print("Error updating count:", e)
    
    def load_model(self):
        try:
            # Load a model trained to detect cement bags
            model_path = get_model_path(r"gui\counter_models\ayub_model.pt")
            print(f"Attempting to load model from: {model_path}")
            
            if os.path.exists(model_path):
                print("Model file exists, loading...")
                self.model = YOLO(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
                print("Model loaded successfully")
                print(f"Model device: {next(self.model.parameters()).device}")
            else:
                print(f"Model file not found, using default YOLOv8 model")
                self.model = YOLO('yolov8n.pt').to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            print(f"Error loading model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            
    def detect_and_count(self, frame, model, region_points, track_history, cam_idx, frame_number=0):
        """Detect and count cement bags in the frame"""
        print(f"Starting detection for camera {cam_idx}")
        print(f"Frame shape: {frame.shape}")
        print(f"Region points: {region_points}")
        
        if frame is None:
            print("Error: Empty frame received")
            return None
            
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print("Error: Frame has incorrect dimensions")
            return None
            
        try:
            results = model.track(frame, persist=True, verbose=False, conf=0.5)
        except Exception as e:
            print(f"Error in model detection: {e}")
            return frame
            
        annotated_frame = frame.copy()

        # Track memory: {track_id: {'counted': bool, 'was_inside': bool, 'last_seen': int}}
        if not hasattr(self, 'track_memory'):
            self.track_memory = [dict(), dict()]

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()

            # Filter for cement bags (assuming class 0 is cement bags)
            cement_indices = [i for i, cls_id in enumerate(class_ids) if cls_id == 0]
            print(f"Found {len(cement_indices)} cement bags in frame")

            for i in cement_indices:
                box = boxes[i]
                track_id = track_ids[i]

                x, y, w, h = box
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)

                # Skip small noise
                if w * h < 1500:
                    continue

                # Use top-right corner for counting
                top_right = (x2, y1)

                # Check if inside region
                is_inside = self.point_in_polygon(top_right, region_points)

                # Initialize track memory for new IDs
                if track_id not in self.track_memory[cam_idx]:
                    self.track_memory[cam_idx][track_id] = {
                        'counted': False,
                        'was_inside': False,
                        'last_seen': frame_number
                    }

                # Get previous inside status
                was_inside = self.track_memory[cam_idx][track_id]['was_inside']

                # Count only if moved from outside -> inside and not already counted
                if not was_inside and is_inside and not self.track_memory[cam_idx][track_id]['counted']:
                    self.count_out[cam_idx] += 1
                    self.counted_bags[cam_idx].add(track_id)
                    self.track_memory[cam_idx][track_id]['counted'] = True
                    self.update_count_labels()
                    print(f"Counted bag ID {track_id} for camera {cam_idx}. New count: {self.count_out[cam_idx]}")

                # Update memory
                self.track_memory[cam_idx][track_id]['was_inside'] = is_inside
                self.track_memory[cam_idx][track_id]['last_seen'] = frame_number

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
        cv2.putText(annotated_frame, f"Cam {cam_idx+1} Count: {self.count_out[cam_idx]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated_frame, current_time, (annotated_frame.shape[1] - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Clean old IDs from memory (avoid ID reuse problems)
        MAX_TRACK_AGE = 50  # frames
        self.track_memory[cam_idx] = {
            tid: mem for tid, mem in self.track_memory[cam_idx].items()
            if frame_number - mem['last_seen'] < MAX_TRACK_AGE
        }

        if self.video_writer[cam_idx]:
            self.video_writer[cam_idx].write(annotated_frame)

        return annotated_frame

    def closeEvent(self, event):
        """Handle application close event"""
        # Save region points if they exist
        if any(self.region_points) and any(self.region_fixed):
            self.save_region_points()
            
        # Clean up resources
        for cam_idx in [0, 1]:
            if self.video_streams[cam_idx]:
                self.video_streams[cam_idx].stop_stream()
            if self.video_writer[cam_idx]:
                self.video_writer[cam_idx].release()
        self.timer.stop()
        event.accept()

    def keyPressEvent(self, event):
        """Handle key press events"""
        # ESC key to exit fullscreen
        if event.key() == Qt.Key.Key_Escape and self.fullscreen_camera is not None:
            self.exit_fullscreen()
        else:
            super().keyPressEvent(event)


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
