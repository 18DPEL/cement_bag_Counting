import sys
import cv2
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QSizePolicy,
    QVBoxLayout, QFileDialog, QHBoxLayout, QComboBox,
    QFrame, QGroupBox, QMessageBox, QSlider, QDoubleSpinBox,
    QLineEdit, QTabWidget, QRadioButton, QButtonGroup, QSplitter,
    QTimeEdit
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QTime
from ultralytics import YOLO
import requests
from pathlib import Path
import numpy as np
import torch
from datetime import datetime


class ResizableVideoLabel(QLabel):
    doubleClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("color: white; font-size: 16px;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap = None  # Store the pixmap separately


    def resizeEvent(self, event):
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(
                self.width(), self.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
        super().resizeEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

    def setPixmap(self, pixmap):
        """Override setPixmap to store the original pixmap"""
        self._pixmap = pixmap
        super().setPixmap(pixmap.scaled(
            self.width(), self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))


class YOLOVideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("    YOLO VIDEO PROCESSING AND CLASS-WISE DETECTION AND VIDEO ANALYSIS APPLICATION DESIGN BY AYUB AHMAD ")
        self.setGeometry(100, 100, 1200, 800)

        self.video_path = None
        self.image_path = None
        self.model = None
        self.selected_class = None
        self.cap = None
        self.processing = False
        self.class_names = []
        self.current_model_name = "No model loaded"
        self.task_type = "detection"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Detection parameters
        self.confidence = 0.5
        self.persist = False
        self.tracker_type = "bytetrack.yaml"
        
        # Available trackers
        self.trackers = {
            "ByteTrack": "bytetrack.yaml",
            "Bot-SORT": "botsort.yaml",
            "None": None
        }
        
        # Model directory setup
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Pretrained model options
        self.pretrained_models = {
            "YOLOv8n": "yolov8n.pt",
            "YOLOv8s": "yolov8s.pt",
            "YOLOv8m": "yolov8m.pt",
            "YOLOv8l": "yolov8l.pt",
            "YOLOv8x": "yolov8x.pt",
            "YOLOv8n-seg": "yolov8n-seg.pt",
            "YOLOv8s-seg": "yolov8s-seg.pt",
            "YOLOv8m-seg": "yolov8m-seg.pt",
            "YOLOv8x-seg": "yolov8x-seg.pt",
            "YOLOv9t": "yolov9t.pt",
            "YOLOv9s": "yolov9c.pt",
            "YOLOv9m": "yolov9m.pt",
            "YOLOv9c": "yolov9c.pt",
            "YOLOv9e": "yolov9e.pt",
            "YOLOv9c-seg": "yolov9c-seg.pt",
            "YOLOv9e-seg": "yolov9e-seg.pt",
            "YOLOv10n": "yolov10n.pt",
            "YOLOv10s": "yolov10s.pt",
            "YOLOv10m": "yolov10m.pt",
            "YOLOv10b": "yolov10b.pt",
            "YOLOv10l": "yolov10l.pt",
            "YOLOv10x": "yolov10x.pt",
            "YOLO11n": "yolo11n.pt",
            "YOLO11s": "yolo11s.pt",
            "YOLO11m": "yolo11m.pt",
            "YOLO11l": "yolo11l.pt",
            "YOLO11x": "yolo11x.pt",
            "YOLO11n-seg": "yolo11-seg.pt",
            "YOLO11s-seg": "yolo11s-seg.pt",
            "YOLO11m-seg": "yolo11m-seg.pt",
            "YOLO11l-seg": "yolo11l-seg.pt",
            "YOLO11x-seg": "yolo11x-seg.pt",
            "YOLO12n": "yolo12n.pt",
            "YOLO12s": "yolo12s.pt",
            "YOLO12m": "yolo12m.pt",
            "YOLO12l": "yolo12l.pt",
            "YOLO12x": "yolo12x.pt"
        }
        self.pretrained_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # Video trimming specific timer
        self.trim_timer = QTimer()
        self.trim_timer.timeout.connect(self.update_trim_frame)
        self.trim_playing = False
        self.trim_video_path = None
        self.trim_cap = None
        self.trim_frame_pos = 0
        self.trim_total_frames = 0
        
        # Frame extraction specific timer
        self.extract_timer = QTimer()
        self.extract_timer.timeout.connect(self.update_extract_frame)
        self.extract_playing = False
        self.extract_video_path = None
        self.extract_cap = None
        self.extract_frame_pos = 0
        self.extract_total_frames = 0
        self.extract_fps = 0

    def init_ui(self):
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(5)
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background: #3d3d3d;
            }
        """)

        # Left panel (controls)
        left_panel = QFrame()
        left_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 5px;")
        left_panel.setMinimumWidth(200)
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(15)

        # Loading Panel
        loading_group = QGroupBox("Loading Panel")
        loading_group.setStyleSheet("""
            QGroupBox {
                background: #3d3d3d;
                border: 2px solid #4d4d4d;
                border-radius: 5px;
                margin-top: 10px;
                color: white;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #00b4d8;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)
        loading_layout = QVBoxLayout()

        # Create a tab widget for different media sources
        source_tabs = QTabWidget()
        source_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #4d4d4d;
                border-radius: 3px;
                background: #3d3d3d;
            }
            QTabBar::tab {
                background: #5d5d5d;
                color: white;
                padding: 5px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
            }
            QTabBar::tab:selected {
                background: #5e548e;
            }
        """)

        # File Tab
        file_tab = QWidget()
        file_layout = QVBoxLayout()
        
        # Video Load Section
        video_load_group = QGroupBox("Video File")
        video_load_group.setStyleSheet("""
            QGroupBox {
                background: #3d3d3d;
                border: 1px solid #4d4d4d;
                border-radius: 5px;
                color: white;
            }
            QGroupBox::title {
                color: #a7c4bc;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)
        video_load_layout = QVBoxLayout()
        
        self.load_video_btn = QPushButton("📁 Load Video File")
        self.load_video_btn.setStyleSheet("""
            QPushButton {
                background-color: #5e548e;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9f86c0;
            }
        """)
        self.load_video_btn.clicked.connect(self.load_video)
        video_load_layout.addWidget(self.load_video_btn)
        video_load_group.setLayout(video_load_layout)
        file_layout.addWidget(video_load_group)
        
        # Image Load Section
        image_load_group = QGroupBox("Image File")
        image_load_group.setStyleSheet(video_load_group.styleSheet())
        image_load_layout = QVBoxLayout()
        
        self.load_image_btn = QPushButton("🖼️ Load Image File")
        self.load_image_btn.setStyleSheet(self.load_video_btn.styleSheet())
        self.load_image_btn.clicked.connect(self.load_image)
        image_load_layout.addWidget(self.load_image_btn)
        
        self.process_image_btn = QPushButton("⚙️ Process Image")
        self.process_image_btn.setStyleSheet("""
            QPushButton {
                background-color: #457b9d;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6a9bc5;
            }
            QPushButton:disabled {
                background-color: #5d5d5d;
            }
        """)
        self.process_image_btn.setEnabled(False)
        self.process_image_btn.clicked.connect(self.process_image)
        image_load_layout.addWidget(self.process_image_btn)
        
        image_load_group.setLayout(image_load_layout)
        file_layout.addWidget(image_load_group)
        
        file_layout.addStretch()
        file_tab.setLayout(file_layout)

        # RTSP Stream Tab
        rtsp_tab = QWidget()
        rtsp_layout = QVBoxLayout()
        
        rtsp_group = QGroupBox("RTSP Stream")
        rtsp_group.setStyleSheet(video_load_group.styleSheet())
        rtsp_inner_layout = QVBoxLayout()
        
        self.rtsp_label = QLabel("RTSP URL:")
        self.rtsp_label.setStyleSheet("color: white;")
        rtsp_inner_layout.addWidget(self.rtsp_label)
        
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("rtsp://username:password@ip:port/stream")
        self.rtsp_input.setStyleSheet("""
            QLineEdit {
                background: #4d4d4d;
                color: white;
                padding: 5px;
                border: 1px solid #5d5d5d;
                border-radius: 4px;
            }
        """)
        rtsp_inner_layout.addWidget(self.rtsp_input)
        
        self.connect_rtsp_btn = QPushButton("🔌 Connect to RTSP")
        self.connect_rtsp_btn.setStyleSheet("""
            QPushButton {
                background-color: #5e548e;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9f86c0;
            }
        """)
        self.connect_rtsp_btn.clicked.connect(self.connect_rtsp)
        rtsp_inner_layout.addWidget(self.connect_rtsp_btn)
        
        # RTSP test button
        self.test_rtsp_btn = QPushButton("🔍 Test RTSP Connection")
        self.test_rtsp_btn.setStyleSheet("""
            QPushButton {
                background-color: #457b9d;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #6a9bc5;
            }
        """)
        self.test_rtsp_btn.clicked.connect(self.test_rtsp_connection)
        rtsp_inner_layout.addWidget(self.test_rtsp_btn)
        
        rtsp_group.setLayout(rtsp_inner_layout)
        rtsp_layout.addWidget(rtsp_group)
        rtsp_layout.addStretch()
        rtsp_tab.setLayout(rtsp_layout)

        # Video Trimming Tab
        trim_tab = QWidget()
        trim_layout = QVBoxLayout()
        
        trim_group = QGroupBox("Video Trimming")
        trim_group.setStyleSheet(video_load_group.styleSheet())
        trim_inner_layout = QVBoxLayout()
        
        # Load video for trimming
        self.load_trim_video_btn = QPushButton("📁 Load Video for Trimming")
        self.load_trim_video_btn.setStyleSheet(self.load_video_btn.styleSheet())
        self.load_trim_video_btn.clicked.connect(self.load_video_for_trimming)
        trim_inner_layout.addWidget(self.load_trim_video_btn)
        
        # Video info display
        self.trim_video_info = QLabel("No video loaded")
        self.trim_video_info.setStyleSheet("color: white;")
        trim_inner_layout.addWidget(self.trim_video_info)
        
        # Current time display for trimming tab
        self.trim_time_label = QLabel("Current Time: 00:00:00")
        self.trim_time_label.setStyleSheet("color: white;")
        trim_inner_layout.addWidget(self.trim_time_label)

        # Time selection
        time_group = QGroupBox("Trim Settings")
        time_group.setStyleSheet("""
            QGroupBox {
                background: #3d3d3d;
                border: 1px solid #4d4d4d;
                border-radius: 5px;
                color: white;
            }
            QGroupBox::title {
                color: #a7c4bc;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)
        time_layout = QVBoxLayout()
        
        # Start time
        start_time_layout = QHBoxLayout()
        start_time_label = QLabel("Start Time:")
        start_time_label.setStyleSheet("color: white;")
        start_time_layout.addWidget(start_time_label)
        
        self.start_time_edit = QTimeEdit()
        self.start_time_edit.setDisplayFormat("HH:mm:ss")
        self.start_time_edit.setTime(QTime(0, 0, 0))
        self.start_time_edit.setStyleSheet("""
            QTimeEdit {
                background: #4d4d4d;
                color: white;
                padding: 5px;
                border: 1px solid #5d5d5d;
                border-radius: 4px;
            }
        """)
        start_time_layout.addWidget(self.start_time_edit)
        time_layout.addLayout(start_time_layout)
        
        # End time
        end_time_layout = QHBoxLayout()
        end_time_label = QLabel("End Time:")
        end_time_label.setStyleSheet("color: white;")
        end_time_layout.addWidget(end_time_label)
        
        self.end_time_edit = QTimeEdit()
        self.end_time_edit.setDisplayFormat("HH:mm:ss")
        self.end_time_edit.setTime(QTime(0, 1, 0))  # Default to 1 minute
        self.end_time_edit.setStyleSheet(self.start_time_edit.styleSheet())
        end_time_layout.addWidget(self.end_time_edit)
        time_layout.addLayout(end_time_layout)
        
        time_group.setLayout(time_layout)
        trim_inner_layout.addWidget(time_group)
        
        # Playback controls
        playback_group = QGroupBox("Playback Controls")
        playback_group.setStyleSheet(time_group.styleSheet())
        playback_layout = QHBoxLayout()
        
        # Play/Pause button
        self.play_pause_btn = QPushButton("⏵")
        self.play_pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #5e548e;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 16px;
                min-width: 40px;
            }
            QPushButton:hover {
                background-color: #9f86c0;
            }
            QPushButton:disabled {
                background-color: #5d5d5d;
            }
        """)
        self.play_pause_btn.setEnabled(False)
        self.play_pause_btn.clicked.connect(self.toggle_trim_playback)
        playback_layout.addWidget(self.play_pause_btn)
        
        # Rewind button
        self.rewind_btn = QPushButton("⏮")
        self.rewind_btn.setStyleSheet(self.play_pause_btn.styleSheet())
        self.rewind_btn.setEnabled(False)
        self.rewind_btn.clicked.connect(self.rewind_video)
        playback_layout.addWidget(self.rewind_btn)
        
        # Forward button
        self.forward_btn = QPushButton("⏭")
        self.forward_btn.setStyleSheet(self.play_pause_btn.styleSheet())
        self.forward_btn.setEnabled(False)
        self.forward_btn.clicked.connect(self.forward_video)
        playback_layout.addWidget(self.forward_btn)
        
        playback_group.setLayout(playback_layout)
        trim_inner_layout.addWidget(playback_group)
        
        # Trim button
        self.trim_btn = QPushButton("✂️ Trim Video")
        self.trim_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a9d8f;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3ab7a8;
            }
            QPushButton:disabled {
                background-color: #5d5d5d;
            }
        """)
        self.trim_btn.setEnabled(False)
        self.trim_btn.clicked.connect(self.trim_video)
        trim_inner_layout.addWidget(self.trim_btn)
        
        trim_group.setLayout(trim_inner_layout)
        trim_layout.addWidget(trim_group)
        trim_layout.addStretch()
        trim_tab.setLayout(trim_layout)

        # Frame Extractor Tab
        extract_tab = QWidget()
        extract_layout = QVBoxLayout()
        
        extract_group = QGroupBox("Frame Extractor")
        extract_group.setStyleSheet(video_load_group.styleSheet())
        extract_inner_layout = QVBoxLayout()
        
        # Load video for frame extraction
        self.load_extract_video_btn = QPushButton("📁 Load Video for Frame Extraction")
        self.load_extract_video_btn.setStyleSheet(self.load_video_btn.styleSheet())
        self.load_extract_video_btn.clicked.connect(self.load_video_for_extraction)
        extract_inner_layout.addWidget(self.load_extract_video_btn)
        
        # Video info display
        self.extract_video_info = QLabel("No video loaded")
        self.extract_video_info.setStyleSheet("color: white;")
        extract_inner_layout.addWidget(self.extract_video_info)
        
        # Current time display
        self.extract_time_label = QLabel("Current Time: 00:00:00")
        self.extract_time_label.setStyleSheet("color: white;")
        extract_inner_layout.addWidget(self.extract_time_label)
        
        # Time selection
        extract_time_group = QGroupBox("Extraction Settings")
        extract_time_group.setStyleSheet(time_group.styleSheet())
        extract_time_layout = QVBoxLayout()
        
        # Start time
        extract_start_time_layout = QHBoxLayout()
        extract_start_time_label = QLabel("Start Time:")
        extract_start_time_label.setStyleSheet("color: white;")
        extract_start_time_layout.addWidget(extract_start_time_label)
        
        self.extract_start_time_edit = QTimeEdit()
        self.extract_start_time_edit.setDisplayFormat("HH:mm:ss")
        self.extract_start_time_edit.setTime(QTime(0, 0, 0))
        self.extract_start_time_edit.setStyleSheet(self.start_time_edit.styleSheet())
        extract_start_time_layout.addWidget(self.extract_start_time_edit)
        extract_time_layout.addLayout(extract_start_time_layout)
        
        # End time
        extract_end_time_layout = QHBoxLayout()
        extract_end_time_label = QLabel("End Time:")
        extract_end_time_label.setStyleSheet("color: white;")
        extract_end_time_layout.addWidget(extract_end_time_label)
        
        self.extract_end_time_edit = QTimeEdit()
        self.extract_end_time_edit.setDisplayFormat("HH:mm:ss")
        self.extract_end_time_edit.setTime(QTime(0, 1, 0))  # Default to 1 minute
        self.extract_end_time_edit.setStyleSheet(self.start_time_edit.styleSheet())
        extract_end_time_layout.addWidget(self.extract_end_time_edit)
        extract_time_layout.addLayout(extract_end_time_layout)
        
        extract_time_group.setLayout(extract_time_layout)
        extract_inner_layout.addWidget(extract_time_group)
        
        # Playback controls
        extract_playback_group = QGroupBox("Playback Controls")
        extract_playback_group.setStyleSheet(time_group.styleSheet())
        extract_playback_layout = QHBoxLayout()
        
        # Play/Pause button
        self.extract_play_pause_btn = QPushButton("⏵")
        self.extract_play_pause_btn.setStyleSheet(self.play_pause_btn.styleSheet())
        self.extract_play_pause_btn.setEnabled(False)
        self.extract_play_pause_btn.clicked.connect(self.toggle_extract_playback)
        extract_playback_layout.addWidget(self.extract_play_pause_btn)
        
        # Rewind button
        self.extract_rewind_btn = QPushButton("⏮")
        self.extract_rewind_btn.setStyleSheet(self.play_pause_btn.styleSheet())
        self.extract_rewind_btn.setEnabled(False)
        self.extract_rewind_btn.clicked.connect(self.extract_rewind_video)
        extract_playback_layout.addWidget(self.extract_rewind_btn)
        
        # Forward button
        self.extract_forward_btn = QPushButton("⏭")
        self.extract_forward_btn.setStyleSheet(self.play_pause_btn.styleSheet())
        self.extract_forward_btn.setEnabled(False)
        self.extract_forward_btn.clicked.connect(self.extract_forward_video)
        extract_playback_layout.addWidget(self.extract_forward_btn)
        
        extract_playback_group.setLayout(extract_playback_layout)
        extract_inner_layout.addWidget(extract_playback_group)
        
        # Extract frames button
        self.extract_frames_btn = QPushButton("📸 Extract Frames")
        self.extract_frames_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a9d8f;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3ab7a8;
            }
            QPushButton:disabled {
                background-color: #5d5d5d;
            }
        """)
        self.extract_frames_btn.setEnabled(False)
        self.extract_frames_btn.clicked.connect(self.extract_frames)
        extract_inner_layout.addWidget(self.extract_frames_btn)
        
        extract_group.setLayout(extract_inner_layout)
        extract_layout.addWidget(extract_group)
        extract_layout.addStretch()
        extract_tab.setLayout(extract_layout)

        # Add tabs to the tab widget
        source_tabs.addTab(file_tab, "File")
        source_tabs.addTab(rtsp_tab, "RTSP Stream")
        source_tabs.addTab(trim_tab, "Video Trimmer")
        source_tabs.addTab(extract_tab, "Frame Extractor")
        loading_layout.addWidget(source_tabs)

        # Model Load Section
        model_load_group = QGroupBox("Model Selection")
        model_load_group.setStyleSheet(video_load_group.styleSheet())
        model_load_layout = QVBoxLayout()
        
        # Current Model Display
        self.current_model_label = QLabel(f"Current Model: {self.current_model_name}")
        self.current_model_label.setStyleSheet("color: #a7c4bc; font-weight: bold;")
        model_load_layout.addWidget(self.current_model_label)
        
        # Model type selection
        model_type_layout = QHBoxLayout()
        self.custom_model_btn = QPushButton("Custom Model")
        self.custom_model_btn.setStyleSheet("""
            QPushButton {
                background-color: #5e548e;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9f86c0;
            }
        """)
        self.custom_model_btn.clicked.connect(self.load_custom_model)
        
        self.pretrained_model_btn = QPushButton("Pretrained Model")
        self.pretrained_model_btn.setStyleSheet(self.custom_model_btn.styleSheet())
        self.pretrained_model_btn.clicked.connect(self.show_pretrained_options)
        model_type_layout.addWidget(self.custom_model_btn)
        model_type_layout.addWidget(self.pretrained_model_btn)
        model_load_layout.addLayout(model_type_layout)
        
        # Pretrained model dropdown (initially hidden)
        self.pretrained_dropdown = QComboBox()
        self.pretrained_dropdown.setPlaceholderText("Select pretrained model")
        self.pretrained_dropdown.addItems(self.pretrained_models.keys())
        self.pretrained_dropdown.setStyleSheet("""
            QComboBox {
                background: #4d4d4d;
                color: white;
                padding: 5px;
                border: 1px solid #5d5d5d;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        self.pretrained_dropdown.currentIndexChanged.connect(self.load_pretrained_model)
        self.pretrained_dropdown.hide()
        model_load_layout.addWidget(self.pretrained_dropdown)
        
        model_load_group.setLayout(model_load_layout)
        loading_layout.addWidget(model_load_group)

        # Class Selection Section
        class_group = QGroupBox("Class Selection")
        class_group.setStyleSheet(video_load_group.styleSheet())
        class_layout = QVBoxLayout()
        
        self.class_dropdown = QComboBox()
        self.class_dropdown.setPlaceholderText("Select a class")
        self.class_dropdown.setEnabled(False)
        self.class_dropdown.setStyleSheet("""
            QComboBox {
                background: #4d4d4d;
                color: white;
                padding: 5px;
                border: 1px solid #5d5d5d;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        self.class_dropdown.currentIndexChanged.connect(self.select_class)
        class_layout.addWidget(self.class_dropdown)
        class_group.setLayout(class_layout)
        loading_layout.addWidget(class_group)

        loading_group.setLayout(loading_layout)
        left_layout.addWidget(loading_group)

        # Control Panel
        control_group = QGroupBox("Control Panel")
        control_group.setStyleSheet(loading_group.styleSheet())
        control_layout = QVBoxLayout()

        # Status Display
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("color: #a7c4bc; font-weight: bold;")
        control_layout.addWidget(self.status_label)

        # Control Buttons
        btn_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("▶ Start Processing")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a9d8f;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3ab7a8;
            }
            QPushButton:disabled {
                background-color: #5d5d5d;
            }
        """)
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_processing)
        
        self.stop_btn = QPushButton("⏹ Stop Processing")
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e76f51;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #f28482;
            }
            QPushButton:disabled {
                background-color: #5d5d5d;
            }
        """)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_processing)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        control_layout.addLayout(btn_layout)

        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        left_layout.addStretch()

        left_panel.setLayout(left_layout)
        main_splitter.addWidget(left_panel)

        # Middle panel (video display)
        middle_panel = QFrame()
        middle_panel.setStyleSheet("background-color: #1e1e1e; border-radius: 5px;")
        middle_layout = QVBoxLayout()
        middle_layout.setContentsMargins(5, 5, 5, 5)

        self.video_label = ResizableVideoLabel("No media loaded")
        middle_layout.addWidget(self.video_label)

        middle_panel.setLayout(middle_layout)
        main_splitter.addWidget(middle_panel)

        # Right panel (filters and tracking)
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 5px;")
        right_panel.setMinimumWidth(100)
        right_panel.setMaximumWidth(200)
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(15)

        # Resource Selection Group
        resource_group = QGroupBox("Resource Selection")
        resource_group.setStyleSheet("""
            QGroupBox {
                background: #3d3d3d;
                border: 2px solid #4d4d4d;
                border-radius: 5px;
                margin-top: 10px;
                color: white;
                font-weight: bold;
            }
            QGroupBox::title {
                color: #00b4d8;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
        """)
        resource_layout = QVBoxLayout()
        
        # Device selection
        device_label = QLabel("Processing Device:")
        device_label.setStyleSheet("color: white;")
        resource_layout.addWidget(device_label)
        
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto (Recommended)")
        self.device_combo.addItem("GPU (CUDA)")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            self.device_combo.setItemText(1, f"GPU ({gpu_name})")
        else:
            self.device_combo.setItemText(1, "GPU (Not Available)")
            self.device_combo.model().item(1).setEnabled(False)
        self.device_combo.addItem("CPU")
        
        self.device_combo.setStyleSheet("""
            QComboBox {
                background: #4d4d4d;
                color: white;
                padding: 5px;
                border: 1px solid #5d5d5d;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        self.device_combo.currentIndexChanged.connect(self.update_processing_device)
        resource_layout.addWidget(self.device_combo)
        
        # Add device info label
        self.device_info_label = QLabel()
        self.device_info_label.setStyleSheet("color: #a7c4bc; font-size: 10px;")
        self.update_device_info()
        resource_layout.addWidget(self.device_info_label)
        
        resource_group.setLayout(resource_layout)
        right_layout.addWidget(resource_group)

        # Detection Settings Group
        detection_group = QGroupBox("Detection Settings")
        detection_group.setStyleSheet(resource_group.styleSheet())
        detection_layout = QVBoxLayout()

        # Confidence Threshold
        confidence_layout = QHBoxLayout()
        confidence_label = QLabel("Confidence:")
        confidence_label.setStyleSheet("color: white;")
        confidence_layout.addWidget(confidence_label)
        
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(int(self.confidence * 100))
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        confidence_layout.addWidget(self.confidence_slider)
        
        self.confidence_spinbox = QDoubleSpinBox()
        self.confidence_spinbox.setRange(0.0, 1.0)
        self.confidence_spinbox.setSingleStep(0.01)
        self.confidence_spinbox.setValue(self.confidence)
        self.confidence_spinbox.valueChanged.connect(self.update_confidence_spinbox)
        confidence_layout.addWidget(self.confidence_spinbox)
        
        detection_layout.addLayout(confidence_layout)

        # Persist Checkbox
        self.persist_checkbox = QPushButton("Persist: OFF")
        self.persist_checkbox.setCheckable(True)
        self.persist_checkbox.setStyleSheet("""
            QPushButton {
                background-color: #5e548e;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #2a9d8f;
            }
        """)
        self.persist_checkbox.clicked.connect(self.toggle_persist)
        detection_layout.addWidget(self.persist_checkbox)

        detection_group.setLayout(detection_layout)
        right_layout.addWidget(detection_group)

        # Task Type Group
        task_group = QGroupBox("Task Type")
        task_group.setStyleSheet(resource_group.styleSheet())
        task_layout = QVBoxLayout()
        
        # Task type radio buttons
        self.task_button_group = QButtonGroup()
        
        self.detection_radio = QRadioButton("Detection")
        self.detection_radio.setChecked(True)
        self.detection_radio.setStyleSheet("color: white;")
        self.task_button_group.addButton(self.detection_radio)
        task_layout.addWidget(self.detection_radio)
        
        self.segmentation_radio = QRadioButton("Segmentation")
        self.segmentation_radio.setStyleSheet("color: white;")
        self.segmentation_radio.setEnabled(False)  # Disabled until segmentation model is loaded
        self.task_button_group.addButton(self.segmentation_radio)
        task_layout.addWidget(self.segmentation_radio)
        
        self.task_button_group.buttonClicked.connect(self.update_task_type)
        
        task_group.setLayout(task_layout)
        right_layout.addWidget(task_group)

        # Tracking Settings Group
        tracking_group = QGroupBox("Tracking Settings")
        tracking_group.setStyleSheet(resource_group.styleSheet())
        tracking_layout = QVBoxLayout()

        # Tracker Selection
        tracker_label = QLabel("Tracker:")
        tracker_label.setStyleSheet("color: white;")
        tracking_layout.addWidget(tracker_label)
        
        self.tracker_dropdown = QComboBox()
        self.tracker_dropdown.addItems(self.trackers.keys())
        self.tracker_dropdown.setCurrentText("ByteTrack")
        self.tracker_dropdown.setStyleSheet("""
            QComboBox {
                background: #4d4d4d;
                color: white;
                padding: 5px;
                border: 1px solid #5d5d5d;
                border-radius: 4px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        self.tracker_dropdown.currentIndexChanged.connect(self.update_tracker)
        tracking_layout.addWidget(self.tracker_dropdown)

        tracking_group.setLayout(tracking_layout)
        right_layout.addWidget(tracking_group)

        right_layout.addStretch()
        right_panel.setLayout(right_layout)
        main_splitter.addWidget(right_panel)

        # Set stretch factors to make middle panel more flexible
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setStretchFactor(2, 1)

        main_layout = QVBoxLayout()
        main_layout.addWidget(main_splitter)
        self.setLayout(main_layout)

        # Connect double click signal
        self.video_label.doubleClicked.connect(self.toggle_fullscreen)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def update_device_info(self):
        """Update the device information label"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info_text = f"GPU: {gpu_name}\nVRAM: {vram:.1f}GB"
        else:
            info_text = "No GPU available"
        self.device_info_label.setText(info_text)

    def update_processing_device(self, index):
        """Handle device selection changes"""
        if index == 0:  # Auto
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif index == 1:  # GPU
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                QMessageBox.warning(self, "GPU Not Available", 
                                   "CUDA GPU is not available on this system. Using CPU instead.")
                self.device_combo.setCurrentIndex(2)  # Switch to CPU
                return
        else:  # CPU
            self.device = 'cpu'
        
        self.status_label.setText(f"Status: Processing device set to {self.device.upper()}")
        
        # Reload model with new device if already loaded
        if self.model is not None:
            try:
                model_path = self.model.ckpt_path if hasattr(self.model, 'ckpt_path') else None
                if model_path:
                    self.load_model_file(model_path)
            except Exception as e:
                self.status_label.setText(f"Status: Error switching device - {str(e)}")

    def update_confidence(self, value):
        self.confidence = value / 100.0
        self.confidence_spinbox.setValue(self.confidence)

    def update_confidence_spinbox(self, value):
        self.confidence = value
        self.confidence_slider.setValue(int(self.confidence * 100))

    def toggle_persist(self, checked):
        self.persist = checked
        self.persist_checkbox.setText(f"Persist: {'ON' if checked else 'OFF'}")

    def update_tracker(self, index):
        tracker_name = self.tracker_dropdown.currentText()
        self.tracker_type = self.trackers[tracker_name]

    def update_task_type(self, button):
        """Update the task type based on selected radio button"""
        self.task_type = "detection" if button == self.detection_radio else "segmentation"
        self.status_label.setText(f"Status: Task type set to {self.task_type}")

    def load_video_for_extraction(self):
        """Handle video file loading for frame extraction"""
        video_formats = "*.mp4 *.avi *.mov *.webm *.mkv *.flv *.wmv"
        
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Video File for Frame Extraction", 
            "", 
            f"Video Files ({video_formats})"
        )
        
        if file_name:
            try:
                # Release any existing video capture
                if hasattr(self, 'extract_cap') and self.extract_cap:
                    self.extract_cap.release()
                
                self.extract_cap = cv2.VideoCapture(file_name)
                if not self.extract_cap.isOpened():
                    raise ValueError("Could not open video file")
                
                # Get video properties
                self.extract_fps = self.extract_cap.get(cv2.CAP_PROP_FPS)
                self.extract_total_frames = int(self.extract_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = self.extract_total_frames / self.extract_fps
                
                # Calculate duration in hours, minutes, seconds
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                
                # Update video info label
                self.extract_video_info.setText(
                    f"Video: {os.path.basename(file_name)}\n"
                    f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}\n"
                    f"FPS: {self.extract_fps:.2f}, Frames: {self.extract_total_frames}"
                )
                
                # Set max end time
                max_time = QTime(hours, minutes, seconds)
                self.extract_end_time_edit.setMaximumTime(max_time)
                self.extract_end_time_edit.setTime(max_time if max_time > QTime(0, 1, 0) else QTime(0, 1, 0))
                
                # Store video path and enable controls
                self.extract_video_path = file_name
                self.extract_frames_btn.setEnabled(True)
                self.extract_play_pause_btn.setEnabled(True)
                self.extract_rewind_btn.setEnabled(True)
                self.extract_forward_btn.setEnabled(True)
                
                # Reset frame position
                self.extract_frame_pos = 0
                self.extract_cap.set(cv2.CAP_PROP_POS_FRAMES, self.extract_frame_pos)
                
                # Display first frame
                ret, frame = self.extract_cap.read()
                if ret:
                    self.display_frame(frame)
                    self.update_extract_time_label()
                
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Error Loading Video", 
                    f"Failed to load video file:\n{str(e)}"
                )
                self.extract_video_info.setText("No video loaded")
                self.extract_frames_btn.setEnabled(False)
                self.extract_play_pause_btn.setEnabled(False)
                self.extract_rewind_btn.setEnabled(False)
                self.extract_forward_btn.setEnabled(False)
                if hasattr(self, 'extract_cap') and self.extract_cap:
                    self.extract_cap.release()
                    self.extract_cap = None

    def toggle_extract_playback(self):
        """Toggle video playback for frame extraction"""
        if not self.extract_playing:
            # Start playback
            self.extract_playing = True
            self.extract_play_pause_btn.setText("⏸")
            self.extract_timer.start(int(1000 / self.extract_fps))  # Play at video's native FPS
        else:
            # Pause playback
            self.extract_playing = False
            self.extract_play_pause_btn.setText("⏵")
            self.extract_timer.stop()

    def extract_rewind_video(self):
        """Rewind video by 5 seconds"""
        if self.extract_cap and self.extract_cap.isOpened():
            current_frame = self.extract_cap.get(cv2.CAP_PROP_POS_FRAMES)
            rewind_frames = int(5 * self.extract_fps)  # 5 seconds worth of frames
            new_frame = max(0, current_frame - rewind_frames)
            self.extract_cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            self.update_extract_frame()

    def extract_forward_video(self):
        """Fast forward video by 5 seconds"""
        if self.extract_cap and self.extract_cap.isOpened():
            current_frame = self.extract_cap.get(cv2.CAP_PROP_POS_FRAMES)
            forward_frames = int(5 * self.extract_fps)  # 5 seconds worth of frames
            new_frame = min(self.extract_total_frames - 1, current_frame + forward_frames)
            self.extract_cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            self.update_extract_frame()

    def update_extract_frame(self):
        """Update frame during frame extraction playback"""
        if self.extract_cap and self.extract_cap.isOpened():
            ret, frame = self.extract_cap.read()
            if ret:
                self.display_frame(frame)
                self.update_extract_time_label()
            else:
                # End of video reached
                self.extract_playing = False
                self.extract_play_pause_btn.setText("⏵")
                self.extract_timer.stop()
                self.extract_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to start
                self.update_extract_frame()

    def update_extract_time_label(self):
        """Update the current time label for frame extraction"""
        if self.extract_cap and self.extract_cap.isOpened():
            current_frame = self.extract_cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_time_sec = current_frame / self.extract_fps
            hours = int(current_time_sec // 3600)
            minutes = int((current_time_sec % 3600) // 60)
            seconds = int(current_time_sec % 60)
            self.extract_time_label.setText(f"Current Time: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def extract_frames(self):
        """Extract frames between specified start and end times"""
        if not hasattr(self, 'extract_video_path') or not os.path.exists(self.extract_video_path):
            QMessageBox.warning(self, "Error", "No video loaded for frame extraction")
            return
            
        # Get start and end times
        start_time = self.extract_start_time_edit.time()
        end_time = self.extract_end_time_edit.time()
        
        # Convert QTime to seconds
        start_sec = start_time.hour() * 3600 + start_time.minute() * 60 + start_time.second()
        end_sec = end_time.hour() * 3600 + end_time.minute() * 60 + end_time.second()
        
        if start_sec >= end_sec:
            QMessageBox.warning(self, "Error", "Start time must be before end time")
            return
            
        # Get output directory
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for Frames",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not output_dir:
            return  # User cancelled
            
        try:
            # Create a subdirectory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"frames_{timestamp}")
            os.makedirs(output_path, exist_ok=True)
            
            # Open the video file
            cap = cv2.VideoCapture(self.extract_video_path)
            
            # Calculate start and end frame numbers
            start_frame = int(start_sec * self.extract_fps)
            end_frame = int(end_sec * self.extract_fps)
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Process frames
            current_frame = start_frame
            frame_count = 0
            self.status_label.setText("Status: Extracting frames...")
            QApplication.processEvents()  # Update UI
            
            while current_frame <= end_frame and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Save frame
                frame_path = os.path.join(output_path, f"frame_{current_frame:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_count += 1
                current_frame += 1
                
                # Update progress every 10 frames
                if current_frame % 10 == 0:
                    progress = (current_frame - start_frame) / (end_frame - start_frame) * 100
                    self.status_label.setText(f"Status: Extracting frames... {progress:.1f}%")
                    QApplication.processEvents()  # Update UI
            
            # Release resources
            cap.release()
            
            self.status_label.setText(f"Status: Extracted {frame_count} frames to {os.path.basename(output_path)}")
            QMessageBox.information(self, "Success", f"Frame extraction completed!\nSaved {frame_count} frames to:\n{output_path}")
            
        except Exception as e:
            self.status_label.setText(f"Status: Error extracting frames - {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to extract frames:\n{str(e)}")
            
            # Clean up if something went wrong
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if os.path.exists(output_path):
                try:
                    os.rmdir(output_path)  # Remove directory if empty
                except:
                    pass

    def load_video_for_trimming(self):
        """Handle video file loading for trimming"""
        video_formats = "*.mp4 *.avi *.mov *.webm *.mkv *.flv *.wmv"
        
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Video File for Trimming", 
            "", 
            f"Video Files ({video_formats})"
        )
        
        if file_name:
            try:
                # Release any existing video capture
                if hasattr(self, 'trim_cap') and self.trim_cap:
                    self.trim_cap.release()
                
                self.trim_cap = cv2.VideoCapture(file_name)
                if not self.trim_cap.isOpened():
                    raise ValueError("Could not open video file")
                
                # Get video properties
                fps = self.trim_cap.get(cv2.CAP_PROP_FPS)
                self.trim_total_frames = int(self.trim_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = self.trim_total_frames / fps
                
                # Calculate duration in hours, minutes, seconds
                hours = int(duration // 3600)
                minutes = int((duration % 3600) // 60)
                seconds = int(duration % 60)
                
                # Update video info label
                self.trim_video_info.setText(
                    f"Video: {os.path.basename(file_name)}\n"
                    f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}\n"
                    f"FPS: {fps:.2f}, Frames: {self.trim_total_frames}"
                )
                
                # Set max end time
                max_time = QTime(hours, minutes, seconds)
                self.end_time_edit.setMaximumTime(max_time)
                self.end_time_edit.setTime(max_time if max_time > QTime(0, 1, 0) else QTime(0, 1, 0))
                
                # Store video path and enable controls
                self.trim_video_path = file_name
                self.trim_btn.setEnabled(True)
                self.play_pause_btn.setEnabled(True)
                self.rewind_btn.setEnabled(True)
                self.forward_btn.setEnabled(True)
                
                # Reset frame position
                self.trim_frame_pos = 0
                self.trim_cap.set(cv2.CAP_PROP_POS_FRAMES, self.trim_frame_pos)
                
                # Update time label
                self.update_trim_time_label()
                
                # Display first frame
                ret, frame = self.trim_cap.read()
                if ret:
                    self.display_frame(frame)
                
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Error Loading Video", 
                    f"Failed to load video file:\n{str(e)}"
                )
                self.trim_video_info.setText("No video loaded")
                self.trim_btn.setEnabled(False)
                self.play_pause_btn.setEnabled(False)
                self.rewind_btn.setEnabled(False)
                self.forward_btn.setEnabled(False)
                if hasattr(self, 'trim_cap') and self.trim_cap:
                    self.trim_cap.release()
                    self.trim_cap = None

    def toggle_trim_playback(self):
        """Toggle video playback for trimming"""
        if not self.trim_playing:
            # Start playback
            self.trim_playing = True
            self.play_pause_btn.setText("⏸")
            self.trim_timer.start(30)  # ~30ms for ~30fps playback
        else:
            # Pause playback
            self.trim_playing = False
            self.play_pause_btn.setText("⏵")
            self.trim_timer.stop()

    def rewind_video(self):
        """Rewind video by 5 seconds"""
        if self.trim_cap and self.trim_cap.isOpened():
            fps = self.trim_cap.get(cv2.CAP_PROP_FPS)
            current_frame = self.trim_cap.get(cv2.CAP_PROP_POS_FRAMES)
            rewind_frames = int(5 * fps)  # 5 seconds worth of frames
            new_frame = max(0, current_frame - rewind_frames)
            self.trim_cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            self.update_trim_frame()

    def forward_video(self):
        """Fast forward video by 5 seconds"""
        if self.trim_cap and self.trim_cap.isOpened():
            fps = self.trim_cap.get(cv2.CAP_PROP_FPS)
            current_frame = self.trim_cap.get(cv2.CAP_PROP_POS_FRAMES)
            forward_frames = int(5 * fps)  # 5 seconds worth of frames
            new_frame = min(self.trim_total_frames - 1, current_frame + forward_frames)
            self.trim_cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            self.update_trim_frame()

    def update_trim_frame(self):
        """Update frame during trimming playback"""
        if self.trim_cap and self.trim_cap.isOpened():
            ret, frame = self.trim_cap.read()
            if ret:
                self.display_frame(frame)
                self.update_trim_time_label()
            else:
                # End of video reached
                self.trim_playing = False
                self.play_pause_btn.setText("⏵")
                self.trim_timer.stop()
                self.trim_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind to start
                self.update_trim_frame()

    def update_trim_time_label(self):
        """Update the current time label for video trimming"""
        if self.trim_cap and self.trim_cap.isOpened():
            fps = self.trim_cap.get(cv2.CAP_PROP_FPS)
            current_frame = self.trim_cap.get(cv2.CAP_PROP_POS_FRAMES)
            current_time_sec = current_frame / fps
            hours = int(current_time_sec // 3600)
            minutes = int((current_time_sec % 3600) // 60)
            seconds = int(current_time_sec % 60)
            self.trim_time_label.setText(f"Current Time: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def trim_video(self):
        """Trim the video between specified start and end times"""
        if not hasattr(self, 'trim_video_path') or not os.path.exists(self.trim_video_path):
            QMessageBox.warning(self, "Error", "No video loaded for trimming")
            return
            
        # Get start and end times
        start_time = self.start_time_edit.time()
        end_time = self.end_time_edit.time()
        
        # Convert QTime to seconds
        start_sec = start_time.hour() * 3600 + start_time.minute() * 60 + start_time.second()
        end_sec = end_time.hour() * 3600 + end_time.minute() * 60 + end_time.second()
        
        if start_sec >= end_sec:
            QMessageBox.warning(self, "Error", "Start time must be before end time")
            return
            
        # Get output file path
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Trimmed Video",
            "",
            "MP4 Files (*.mp4)"
        )
        
        if not output_path:
            return  # User cancelled
            
        # Ensure .mp4 extension
        if not output_path.lower().endswith('.mp4'):
            output_path += '.mp4'
            
        try:
            # Open the video file
            cap = cv2.VideoCapture(self.trim_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate start and end frame numbers
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Process frames
            current_frame = start_frame
            self.status_label.setText("Status: Trimming video...")
            QApplication.processEvents()  # Update UI
            
            while current_frame <= end_frame and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                out.write(frame)
                current_frame += 1
                
                # Update progress every 10 frames
                if current_frame % 10 == 0:
                    progress = (current_frame - start_frame) / (end_frame - start_frame) * 100
                    self.status_label.setText(f"Status: Trimming video... {progress:.1f}%")
                    QApplication.processEvents()  # Update UI
            
            # Release resources
            cap.release()
            out.release()
            
            self.status_label.setText(f"Status: Video trimmed and saved to {os.path.basename(output_path)}")
            QMessageBox.information(self, "Success", "Video trimming completed successfully!")
            
        except Exception as e:
            self.status_label.setText(f"Status: Error trimming video - {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to trim video:\n{str(e)}")
            
            # Clean up if something went wrong
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'out' in locals() and out.isOpened():
                out.release()
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass

    def load_video(self):
        # Define supported video formats
        video_formats = "*.mp4 *.avi *.mov *.webm *.mkv *.flv *.wmv"
        
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Video File", 
            "", 
            f"Video Files ({video_formats})"
        )
        
        if file_name:
            # Validate file extension
            valid_extensions = [ext[1:] for ext in video_formats.split()]
            file_ext = os.path.splitext(file_name)[1].lower()
            
            if file_ext not in valid_extensions:
                QMessageBox.warning(
                    self, 
                    "Invalid File", 
                    f"Please select a valid video file.\nSupported formats: {', '.join(valid_extensions)}"
                )
                return
            
            # Try to open the video file to verify it's actually a video
            try:
                cap = cv2.VideoCapture(file_name)
                if not cap.isOpened():
                    raise ValueError("Could not open video file")
                
                # Release the test capture and create a new one
                cap.release()
                self.cap = cv2.VideoCapture(file_name)
                
                self.video_path = file_name
                self.image_path = None  # Clear any loaded image
                if self.model:
                    self.start_btn.setEnabled(True)
                    self.process_image_btn.setEnabled(False)
                self.status_label.setText(f"Status: Video loaded - {os.path.basename(file_name)}")
                
                # Display first frame
                ret, frame = self.cap.read()
                if ret:
                    self.display_frame(frame)
                
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Error Loading Video", 
                    f"Failed to load video file:\n{str(e)}"
                )
                if hasattr(self, 'cap') and self.cap:
                    self.cap.release()
                    self.cap = None
                self.status_label.setText("Status: Failed to load video")

    def load_image(self):
        """Handle image file loading"""
        image_formats = "*.jpg *.jpeg *.png *.bmp *.tiff"
        
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image File",
            "",
            f"Image Files ({image_formats})"
        )
        
        if file_name:
            try:
                # Validate file extension
                valid_extensions = [ext[1:] for ext in image_formats.split()]
                file_ext = os.path.splitext(file_name)[1].lower()
                
                if file_ext not in valid_extensions:
                    QMessageBox.warning(
                        self,
                        "Invalid File",
                        f"Please select a valid image file.\nSupported formats: {', '.join(valid_extensions)}"
                    )
                    return
                
                # Load and display the image
                image = cv2.imread(file_name)
                if image is None:
                    raise ValueError("Could not open image file")
                
                self.image_path = file_name
                self.video_path = None  # Clear any loaded video
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
                if self.model:
                    self.process_image_btn.setEnabled(True)
                    self.start_btn.setEnabled(False)
                
                self.status_label.setText(f"Status: Image loaded - {os.path.basename(file_name)}")
                self.display_frame(image)
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error Loading Image",
                    f"Failed to load image file:\n{str(e)}"
                )
                self.status_label.setText("Status: Failed to load image")

    def process_image(self):
        """Process the loaded image with the current model and settings"""
        if not self.image_path or not os.path.exists(self.image_path):
            self.status_label.setText("Status: Error - No image loaded or image file missing!")
            return
            
        if not self.model:
            self.status_label.setText("Status: Error - Please load a YOLO model first!")
            return
            
        if self.selected_class is None:
            self.status_label.setText("Status: Error - Please select a class first!")
            return
            
        try:
            # Load the image
            frame = cv2.imread(self.image_path)
            if frame is None:
                raise ValueError("Could not read image file")
            
            # Process based on task type
            if self.task_type == "detection":
                results = self.model.predict(
                    frame,
                    conf=self.confidence,
                    classes=[self.selected_class]
                )
                
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        label = f"{self.model.names[cls_id]} {conf:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            elif self.task_type == "segmentation":
                results = self.model.predict(
                    frame,
                    conf=self.confidence,
                    classes=[self.selected_class]
                )
                
                for r in results:
                    # Draw segmentation masks if available
                    if hasattr(r, 'masks') and r.masks is not None:
                        for mask in r.masks:
                            # Get the mask as a numpy array
                            mask_data = mask.data[0].cpu().numpy()
                            mask_data = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]))
                            
                            # Create a colored mask overlay
                            color_mask = (0, 255, 0)  # Green color for segmentation
                            colored_mask = np.zeros_like(frame)
                            colored_mask[:] = color_mask
                            
                            # Apply mask
                            mask_bool = mask_data > 0.5
                            frame[mask_bool] = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)[mask_bool]
                            
                            # Draw bounding box
                            if hasattr(mask, 'boxes') and mask.boxes is not None:
                                box = mask.boxes[0]
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Draw label
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                label = f"{self.model.names[cls_id]} {conf:.2f}"
                                cv2.putText(frame, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            self.display_frame(frame)
            self.status_label.setText(f"Status: Image processed ({self.task_type})")
            
        except Exception as e:
            self.status_label.setText(f"Status: Error processing image - {str(e)}")
            QMessageBox.critical(self, "Processing Error", f"Failed to process image: {str(e)}")

    def connect_rtsp(self):
        """Handle RTSP stream connection"""
        rtsp_url = self.rtsp_input.text().strip()
        if not rtsp_url:
            QMessageBox.warning(self, "Input Error", "Please enter an RTSP URL")
            return
        
        # Test the connection first
        test_cap = cv2.VideoCapture(rtsp_url)
        if not test_cap.isOpened():
            QMessageBox.critical(self, "Connection Error", "Failed to connect to RTSP stream")
            test_cap.release()
            return
        
        # Connection successful, proceed with setup
        test_cap.release()
        
        try:
            # Set up RTSP stream with buffer size adjustment
            self.cap = cv2.VideoCapture(rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for RTSP
            
            if not self.cap.isOpened():
                raise ConnectionError("Failed to open RTSP stream")
            
            self.video_path = rtsp_url
            self.image_path = None  # Clear any loaded image
            if self.model:
                self.start_btn.setEnabled(True)
                self.process_image_btn.setEnabled(False)
            self.status_label.setText(f"Status: Connected to RTSP stream")
            
            # Display first frame
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                
        except Exception as e:
            QMessageBox.critical(self, "RTSP Error", f"Failed to establish RTSP connection:\n{str(e)}")
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.cap = None
            self.status_label.setText("Status: RTSP connection failed")

    def test_rtsp_connection(self):
        """Test the RTSP connection without setting it up for processing"""
        rtsp_url = self.rtsp_input.text().strip()
        if not rtsp_url:
            QMessageBox.warning(self, "Input Error", "Please enter an RTSP URL")
            return
        
        self.status_label.setText("Status: Testing RTSP connection...")
        QApplication.processEvents()  # Update UI
        
        try:
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                raise ConnectionError("Failed to open RTSP stream")
            
            # Try to read a frame
            ret, frame = cap.read()
            if not ret:
                raise ConnectionError("Could not read frame from RTSP stream")
            
            # Display the test frame
            self.display_frame(frame)
            
            QMessageBox.information(self, "Success", "RTSP connection test successful!")
            self.status_label.setText("Status: RTSP test successful")
            
        except Exception as e:
            QMessageBox.critical(self, "Test Failed", f"RTSP connection test failed:\n{str(e)}")
            self.status_label.setText("Status: RTSP test failed")
            
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()

    def display_frame(self, frame):
        """Display a frame in the video label"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)

    def load_custom_model(self):
        """Handle custom model loading"""
        self.pretrained_dropdown.hide()
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Custom YOLO Model File", "", "Model Files (*.pt)")
        if model_path:
            self.load_model_file(model_path)

    def show_pretrained_options(self):
        """Show the pretrained model dropdown"""
        self.pretrained_dropdown.show()

    def load_pretrained_model(self, index):
        """Handle pretrained model selection and downloading"""
        if index >= 0:
            model_name = self.pretrained_dropdown.currentText()
            model_filename = self.pretrained_models[model_name]
            model_path = os.path.join(self.model_dir, model_filename)
            
            # Check if model already exists
            if os.path.exists(model_path):
                self.status_label.setText(f"Status: Loading {model_name}...")
                self.load_model_file(model_path)
            else:
                self.download_pretrained_model(model_name, model_filename)

    def download_pretrained_model(self, model_name, model_filename):
        """Download pretrained model from Ultralytics repository"""
        url = f"{self.pretrained_url}{model_filename}"
        model_path = os.path.join(self.model_dir, model_filename)
        
        self.status_label.setText(f"Status: Downloading {model_name}...")
        QApplication.processEvents()  # Update UI
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.status_label.setText(f"Status: {model_name} downloaded successfully!")
            self.load_model_file(model_path)
            
        except Exception as e:
            self.status_label.setText(f"Status: Error downloading {model_name} - {str(e)}")
            QMessageBox.critical(self, "Download Error", f"Failed to download model: {str(e)}")

    def load_model_file(self, model_path):
        """Load model from file with improved segmentation support"""
        try:
            self.model = YOLO(model_path).to(self.device)
            self.class_names = self.model.names
            
            # Check if model supports segmentation
            self.is_segmentation_model = False
            try:
                # Try to get model task type (newer versions of ultralytics)
                model_task = self.model.task
                self.is_segmentation_model = model_task == 'segment'
            except:
                # Fallback for older versions - check model filename or architecture
                if '-seg' in model_path.lower() or 'segment' in model_path.lower():
                    self.is_segmentation_model = True
            
            self.populate_class_dropdown()
            self.class_dropdown.setEnabled(True)
            
            # Update current model name with device info
            self.current_model_name = os.path.basename(model_path)
            device_info = "GPU" if 'cuda' in str(self.device) else "CPU"
            self.current_model_label.setText(f"Current Model: {self.current_model_name} ({device_info})")
            
            # Enable segmentation radio button if model supports it
            self.segmentation_radio.setEnabled(self.is_segmentation_model)
            if not self.is_segmentation_model:
                self.detection_radio.setChecked(True)
                self.task_type = "detection"
                self.status_label.setText(f"Status: Model loaded on {device_info} - detection only")
            else:
                self.status_label.setText(f"Status: Model loaded on {device_info} - supports segmentation")
            
            # Enable appropriate processing buttons
            if self.video_path and self.cap and self.cap.isOpened():
                self.start_btn.setEnabled(True)
            if self.image_path:
                self.process_image_btn.setEnabled(True)
                
        except Exception as e:
            self.status_label.setText(f"Status: Error loading model - {str(e)}")
            QMessageBox.critical(self, "Load Error", f"Failed to load model: {str(e)}")

    def populate_class_dropdown(self):
        self.class_dropdown.clear()
        if self.class_names:
            for class_id, class_name in self.class_names.items():
                self.class_dropdown.addItem(f"{class_name} (ID: {class_id})", class_id)
            self.class_dropdown.setCurrentIndex(-1)

    def select_class(self, index):
        if index >= 0 and self.class_names:
            self.selected_class = self.class_dropdown.itemData(index)
            class_name = self.class_names[self.selected_class]
            self.status_label.setText(f"Status: Selected class - {class_name} (ID: {self.selected_class})")

    def start_processing(self):
        if not self.cap or not self.cap.isOpened():
            self.status_label.setText("Status: Error - Please load a video first!")
            return
            
        if not self.model:
            self.status_label.setText("Status: Error - Please load a YOLO model first!")
            return
            
        if self.selected_class is None:
            self.status_label.setText("Status: Error - Please select a class first!")
            return
            
        self.processing = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.timer.start(30)
        self.status_label.setText(f"Status: Processing video ({self.task_type})...")

    def stop_processing(self):
        self.processing = False
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Status: Processing stopped")

    def update_frame(self):
        if not self.processing:
            return
            
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                self.timer.stop()
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.status_label.setText("Status: Video ended")
                return

            if self.model and self.selected_class is not None:
                # Prepare tracking arguments if tracker is selected
                tracker_args = None
                if self.tracker_type and self.task_type == "detection":  # Tracking only for detection
                    tracker_args = {
                        "tracker": self.tracker_type,
                        "persist": self.persist
                    }
                
                # Run inference based on selected task type
                if self.task_type == "detection":
                    results = self.model.predict(
                        frame, 
                        conf=self.confidence,
                        classes=[self.selected_class],
                        **({"tracker": tracker_args} if tracker_args else {})
                    )
                    
                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            track_id = int(box.id[0]) if box.id is not None else None
                            
                            label = f"{self.model.names[cls_id]} {conf:.2f}"
                            if track_id is not None:
                                label += f" ID:{track_id}"
                                
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                elif self.task_type == "segmentation":
                    try:
                        results = self.model.predict(
                            frame,
                            conf=self.confidence,
                            classes=[self.selected_class]
                        )
                        
                        for r in results:
                            # Draw segmentation masks if available
                            if hasattr(r, 'masks') and r.masks is not None:
                                for mask in r.masks:
                                    # Get the mask as a numpy array
                                    mask_data = mask.data[0].cpu().numpy()
                                    mask_data = cv2.resize(mask_data, (frame.shape[1], frame.shape[0]))
                                    
                                    # Create a colored mask overlay
                                    color_mask = (0, 255, 0)  # Green color for segmentation
                                    colored_mask = np.zeros_like(frame)
                                    colored_mask[:] = color_mask
                                    
                                    # Apply mask
                                    mask_bool = mask_data > 0.5
                                    frame[mask_bool] = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)[mask_bool]
                                    
                                    # Draw bounding box
                                    if hasattr(mask, 'boxes') and mask.boxes is not None:
                                        box = mask.boxes[0]
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        
                                        # Draw label
                                        cls_id = int(box.cls[0])
                                        conf = float(box.conf[0])
                                        label = f"{self.model.names[cls_id]} {conf:.2f}"
                                        cv2.putText(frame, label, (x1, y1 - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    except Exception as e:
                        print(f"Segmentation error: {str(e)}")
                        self.status_label.setText(f"Status: Segmentation error - {str(e)}")

            self.display_frame(frame)

    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if hasattr(self, 'trim_cap') and self.trim_cap:
            self.trim_cap.release()
        if hasattr(self, 'extract_cap') and self.extract_cap:
            self.extract_cap.release()
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        if hasattr(self, 'trim_timer') and self.trim_timer.isActive():
            self.trim_timer.stop()
        if hasattr(self, 'extract_timer') and self.extract_timer.isActive():
            self.extract_timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOVideoApp()
    window.show()
    sys.exit(app.exec())


