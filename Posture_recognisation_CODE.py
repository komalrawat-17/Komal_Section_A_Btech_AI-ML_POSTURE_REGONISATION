import sys
import cv2
import csv
import os
import time
import threading
import numpy as np
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QWidget, QVBoxLayout,
    QHBoxLayout, QStackedLayout, QListWidget, QTextEdit, QFileDialog,
    QMessageBox, QTabWidget, QGraphicsOpacityEffect, QProgressBar, QSpinBox,
    QDoubleSpinBox, QLineEdit, QCheckBox, QComboBox
)

import pyttsx3
import mediapipe as mp

# Pose detection setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# File paths
LOG_DIR = "logs"
POSTURE_LOG_FILE = os.path.join(LOG_DIR, 'bad_posture_log.csv')
SESSION_LOG_FILE = os.path.join(LOG_DIR, 'posture_session_log.csv')
EXERCISES_FILE = os.path.join(LOG_DIR, 'posture_exercises.txt')

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

class PostureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Posture Corrector Pro")
        self.setGeometry(100, 100, 1100, 750)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video device")
            sys.exit(1)
        
        # Calibration & session variables
        self.total_calibration_frames = 50
        self.calibration_frames = self.total_calibration_frames
        self.calibrated = False
        # For new measurement based on view mode
        self.calibration_measurements = []
        self.calibrated_measurement = None
        
        self.session_active = False
        self.session_start_time = 0
        self.total_frames = 0
        self.good_posture_frames = 0
        self.poor_posture_frames = 0
        self.poor_alerts_count = 0  # Count of posture alerts, for dashboard
        self.last_alert_time = 0
        self.alert_cooldown = 5   # seconds (can be customized via settings)
        
        # Settings defaults
        self.voice_enabled = True
        self.export_folder = ""
        self.view_mode = "Front"  # "Front" or "Side"
        
        # Initialize instance variables for animations
        self.fade_in = None
        self.fade_out = None
        
        # Initialize voice engine
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
        except Exception as e:
            print(f"Voice engine initialization failed: {e}")
            self.engine = None
        
        # Initialize log files
        self.initialize_log_files()
        
        # Setup UI (dashboard, setup, logs, exercises, settings)
        self.setup_ui()
        
        # Start camera update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
    
    def initialize_log_files(self):
        """Initialize log files with headers if they don't exist."""
        if not os.path.exists(POSTURE_LOG_FILE):
            with open(POSTURE_LOG_FILE, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Measurement', 'Posture Status'])
        if not os.path.exists(SESSION_LOG_FILE):
            with open(SESSION_LOG_FILE, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Session Start', 'Duration (s)', 'Good Posture %', 'Bad Posture %'])
        if not os.path.exists(EXERCISES_FILE):
            with open(EXERCISES_FILE, 'w') as file:
                file.write("""Posture Improvement Exercises:
1. Chin Tucks: Gently pull your chin straight back
2. Shoulder Blade Squeeze: Squeeze shoulder blades together
3. Upper Back Stretch: Clasp hands in front and round your upper back
4. Chest Opener: Clasp hands behind your back and lift arms""")
    
    def setup_ui(self):
        """Setup the main UI with enhanced visual elements."""
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)
        
        # Sidebar with fancy, bold fonts
        self.sidebar = QListWidget()
        self.sidebar.addItems(["Dashboard", "Setup", "Logs", "Exercises", "Settings"])
        self.sidebar.setFixedWidth(150)
        self.sidebar.setStyleSheet("""
            QListWidget {
                background-color: #2c3e50;
                color: white;
                font-size: 16px;
                font-family: 'Segoe UI', sans-serif;
                font-weight: bold;
                border: none;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #34495e;
            }
            QListWidget::item:selected {
                background-color: #3498db;
            }
        """)
        self.sidebar.currentRowChanged.connect(self.switch_tab)
        
        # Stacked layout for tabs
        self.stack = QStackedLayout()
        # Create tabs
        self.dashboard_tab = self.create_dashboard_tab()
        self.setup_tab = self.create_setup_tab()
        self.logs_tab = self.create_logs_tab()
        self.exercises_tab = self.create_exercises_tab()
        self.settings_tab = self.create_settings_tab()
        
        # Add tabs to stack
        self.stack.addWidget(self.dashboard_tab)
        self.stack.addWidget(self.setup_tab)
        self.stack.addWidget(self.logs_tab)
        self.stack.addWidget(self.exercises_tab)
        self.stack.addWidget(self.settings_tab)
        
        self.main_layout.addWidget(self.sidebar)
        self.main_layout.addLayout(self.stack)
    
    def switch_tab(self, index):
        """Switch between tabs with a smooth fade transition."""
        new_widget = self.stack.widget(index)
        old_widget = self.stack.currentWidget()
        if (old_widget is None) or (old_widget == new_widget):
            self.stack.setCurrentIndex(index)
            if index == 2:  # Logs tab
                self.load_logs()
            return
        
        self.animate_transition(old_widget, new_widget)
        if index == 2:
            self.load_logs()
    
    def animate_transition(self, old_widget, new_widget):
        """Fade out old widget and fade in new widget."""
        old_effect = QGraphicsOpacityEffect(old_widget)
        old_widget.setGraphicsEffect(old_effect)
        self.fade_out = QPropertyAnimation(old_effect, b"opacity")
        self.fade_out.setDuration(300)
        self.fade_out.setStartValue(1)
        self.fade_out.setEndValue(0)
        self.fade_out.setEasingCurve(QEasingCurve.OutQuad)
        
        new_effect = QGraphicsOpacityEffect(new_widget)
        new_widget.setGraphicsEffect(new_effect)
        new_effect.setOpacity(0)
        self.fade_in = QPropertyAnimation(new_effect, b"opacity")
        self.fade_in.setDuration(300)
        self.fade_in.setStartValue(0)
        self.fade_in.setEndValue(1)
        self.fade_in.setEasingCurve(QEasingCurve.InQuad)
        
        def on_fade_out_finished():
            self.stack.setCurrentWidget(new_widget)
            self.fade_in.start()
        
        self.fade_out.finished.connect(on_fade_out_finished)
        self.fade_out.start()
    
    def create_dashboard_tab(self):
        """Create dashboard tab with video feed, summary cards, score bar, etc."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Summary Cards Layout
        summary_layout = QHBoxLayout()
        card_style = """
            background-color: #ecf0f1;
            border-radius: 10px;
            padding: 15px;
            font-family: 'Segoe UI', sans-serif;
            font-weight: bold;
            font-size: 16px;
            color: #2c3e50;
        """
        self.session_time_card = QLabel("📊 Session Time: 00:00:00")
        self.session_time_card.setStyleSheet(card_style)
        self.good_posture_card = QLabel("✅ Good Posture %: --%")
        self.good_posture_card.setStyleSheet(card_style)
        self.poor_posture_card = QLabel("❌ Poor Posture Alerts: 0")
        self.poor_posture_card.setStyleSheet(card_style)
        summary_layout.addWidget(self.session_time_card)
        summary_layout.addWidget(self.good_posture_card)
        summary_layout.addWidget(self.poor_posture_card)
        layout.addLayout(summary_layout)
        
        # Video display and calibration progress
        self.video_label = QLabel("Starting Camera...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(640, 480)
        layout.addWidget(self.video_label)
        
        self.calibration_progress = QProgressBar()
        self.calibration_progress.setMaximum(self.total_calibration_frames)
        self.calibration_progress.setValue(0)
        self.calibration_progress.setVisible(False)
        layout.addWidget(self.calibration_progress)
        
        # Posture Score Visualization Bar (optional)
        score_layout = QVBoxLayout()
        score_label = QLabel("Posture Score")
        score_label.setAlignment(Qt.AlignCenter)
        score_label.setStyleSheet("font-family: 'Segoe UI', sans-serif; font-weight: bold; font-size: 16px;")
        self.posture_score_bar = QProgressBar()
        self.posture_score_bar.setMaximum(100)
        self.posture_score_bar.setValue(0)
        score_layout.addWidget(score_label)
        score_layout.addWidget(self.posture_score_bar)
        layout.addLayout(score_layout)
        
        # Color Gradient Feedback Bar
        self.feedback_bar = QLabel()
        self.feedback_bar.setFixedHeight(25)
        self.feedback_bar.setStyleSheet("background-color: #2ecc71;")
        layout.addWidget(self.feedback_bar)
        
        # Posture Status Label
        self.posture_status_label = QLabel()
        self.posture_status_label.setAlignment(Qt.AlignCenter)
        self.posture_status_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                font-family: 'Segoe UI', sans-serif;
                padding: 10px;
                border-radius: 5px;
                margin: 10px;
            }
        """)
        layout.addWidget(self.posture_status_label)
        
        return tab
    
    def create_setup_tab(self):
        """Create the setup tab with session controls."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.start_btn = QPushButton("Start Session")
        self.start_btn.clicked.connect(self.start_session)
        self.end_btn = QPushButton("End Session")
        self.end_btn.clicked.connect(self.end_session)
        self.end_btn.setEnabled(False)
        self.calibrate_btn = QPushButton("Calibrate Posture")
        self.calibrate_btn.clicked.connect(self.start_calibration)
        for btn in [self.start_btn, self.end_btn, self.calibrate_btn]:
            btn.setStyleSheet(self.button_style())
            btn.setMinimumHeight(40)
        layout.addWidget(self.calibrate_btn)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.end_btn)
        layout.addStretch()
        return tab
    
    def create_logs_tab(self):
        """Create the logs tab with tabbed interface."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.log_tabs = QTabWidget()
        self.posture_log_viewer = QTextEdit()
        self.posture_log_viewer.setReadOnly(True)
        self.posture_log_viewer.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 12px;
                background-color: #f8f8f8;
            }
        """)
        self.session_log_viewer = QTextEdit()
        self.session_log_viewer.setReadOnly(True)
        self.session_log_viewer.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 12px;
                background-color: #f8f8f8;
            }
        """)
        self.log_tabs.addTab(self.posture_log_viewer, "Posture Events")
        self.log_tabs.addTab(self.session_log_viewer, "Session History")
        btn_layout = QHBoxLayout()
        self.refresh_logs_btn = QPushButton("Refresh Logs")
        self.refresh_logs_btn.clicked.connect(self.load_logs)
        self.export_logs_btn = QPushButton("Export Logs")
        self.export_logs_btn.clicked.connect(self.export_logs)
        for btn in [self.refresh_logs_btn, self.export_logs_btn]:
            btn.setStyleSheet(self.button_style())
            btn_layout.addWidget(btn)
        layout.addWidget(self.log_tabs)
        layout.addLayout(btn_layout)
        return tab
    
    def create_exercises_tab(self):
        """Create the exercises tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.exercises_viewer = QTextEdit()
        self.exercises_viewer.setReadOnly(True)
        self.exercises_viewer.setStyleSheet("font-size: 14px; font-family: 'Segoe UI', sans-serif;")
        try:
            with open(EXERCISES_FILE, 'r') as f:
                self.exercises_viewer.setPlainText(f.read())
        except FileNotFoundError:
            self.exercises_viewer.setPlainText("Exercises file not found.")
        layout.addWidget(self.exercises_viewer)
        return tab
    
    def create_settings_tab(self):
        """Create a settings tab where users can customize parameters."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Alert cooldown setting
        ac_layout = QHBoxLayout()
        ac_label = QLabel("Alert Cooldown (s):")
        ac_label.setStyleSheet("font-family: 'Segoe UI', sans-serif; font-weight: bold;")
        self.ac_spinbox = QSpinBox()
        self.ac_spinbox.setMinimum(1)
        self.ac_spinbox.setMaximum(60)
        self.ac_spinbox.setValue(self.alert_cooldown)
        ac_layout.addWidget(ac_label)
        ac_layout.addWidget(self.ac_spinbox)
        layout.addLayout(ac_layout)
        
        # Voice on/off
        voice_layout = QHBoxLayout()
        voice_label = QLabel("Enable Voice Alerts:")
        voice_label.setStyleSheet("font-family: 'Segoe UI', sans-serif; font-weight: bold;")
        self.voice_checkbox = QCheckBox()
        self.voice_checkbox.setChecked(self.voice_enabled)
        voice_layout.addWidget(voice_label)
        voice_layout.addWidget(self.voice_checkbox)
        layout.addLayout(voice_layout)
        
        # View mode selection
        view_mode_layout = QHBoxLayout()
        view_mode_label = QLabel("View Mode:")
        view_mode_label.setStyleSheet("font-family: 'Segoe UI', sans-serif; font-weight: bold;")
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Front", "Side"])
        self.view_mode_combo.setCurrentText(self.view_mode)
        view_mode_layout.addWidget(view_mode_label)
        view_mode_layout.addWidget(self.view_mode_combo)
        layout.addLayout(view_mode_layout)
        
        # Posture thresholds manual override (if desired)
        pt_layout = QHBoxLayout()
        # Note: With calibration now done on a measurement value, these fields could be repurposed if needed.
        pt_label = QLabel("(Calibration defines your ideal measurement.)")
        pt_label.setStyleSheet("font-family: 'Segoe UI', sans-serif; font-style: italic;")
        pt_layout.addWidget(pt_label)
        layout.addLayout(pt_layout)
        
        # Export folder setting
        ef_layout = QHBoxLayout()
        export_label = QLabel("Default Export Folder:")
        export_label.setStyleSheet("font-family: 'Segoe UI', sans-serif; font-weight: bold;")
        self.export_folder_line = QLineEdit()
        self.export_folder_line.setPlaceholderText("Choose export folder...")
        self.export_folder_line.setText(self.export_folder)
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(self.button_style())
        browse_btn.clicked.connect(self.browse_export_folder)
        ef_layout.addWidget(export_label)
        ef_layout.addWidget(self.export_folder_line)
        ef_layout.addWidget(browse_btn)
        layout.addLayout(ef_layout)
        
        # Save settings button
        save_btn = QPushButton("Save Settings")
        save_btn.setStyleSheet(self.button_style())
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        
        layout.addStretch()
        return tab
    
    def browse_export_folder(self):
        """Open dialog to select an export folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if folder:
            self.export_folder_line.setText(folder)
    
    def save_settings(self):
        """Save settings from the Settings tab."""
        self.alert_cooldown = self.ac_spinbox.value()
        self.voice_enabled = self.voice_checkbox.isChecked()
        self.view_mode = self.view_mode_combo.currentText()
        self.export_folder = self.export_folder_line.text().strip()
        QMessageBox.information(self, "Settings Saved", "Your settings have been saved.")
    
    def button_style(self):
        """Return a consistent, fancy button style."""
        return """
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                font-family: 'Segoe UI', sans-serif;
                font-weight: bold;
                border-radius: 5px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c6da8;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """
    
    def update_frame(self):
        """Update the camera frame, process posture, and update dashboard summaries."""
        try:
            ret, frame = self.cap.read()
            if not ret:
                return
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            height, width, _ = frame.shape
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                def get_coords(landmark):
                    return int(landmark.x * width), int(landmark.y * height)
                
                # Retrieve key landmarks
                left_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER])
                right_shoulder = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER])
                left_hip = get_coords(landmarks[mp_pose.PoseLandmark.LEFT_HIP])
                right_hip = get_coords(landmarks[mp_pose.PoseLandmark.RIGHT_HIP])
                nose_point = get_coords(landmarks[mp_pose.PoseLandmark.NOSE])
                
                # Compute a common midpoint for the shoulders, which we use as the "neck" key point.
                shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2,
                                (left_shoulder[1] + right_shoulder[1]) / 2)
                
                # Depending on view mode, compute a measurement:
                if self.view_mode == "Front":
                    # Using shoulder, neck (midpoint) and face (nose)
                    # We'll consider the horizontal deviation of the nose from the shoulder midpoint.
                    measurement = abs(nose_point[0] - shoulder_mid[0])
                else:  # Side view
                    # For side view, we use hip, neck, and head (nose).
                    hip_mid = ((left_hip[0] + right_hip[0]) / 2,
                               (left_hip[1] + right_hip[1]) / 2)
                    # Calculate the angle at the neck (shoulder midpoint) formed by hip (left side) and nose.
                    def angle(a, b, c):
                        a, b, c = np.array(a), np.array(b), np.array(c)
                        ab = a - b
                        bc = c - b
                        cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
                        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
                    measurement = angle(hip_mid, shoulder_mid, nose_point)
                
                # Calibration: If not calibrated, collect measurements.
                if not self.calibrated and self.calibration_frames > 0:
                    self.calibration_measurements.append(measurement)
                    self.calibration_frames -= 1
                    self.calibration_progress.setVisible(True)
                    self.calibration_progress.setValue(self.total_calibration_frames - self.calibration_frames)
                    if self.calibration_frames == 0:
                        self.calibrated = True
                        self.calibrated_measurement = np.mean(self.calibration_measurements)
                        self.speak_async("Calibration complete. You can now start a session.")
                        self.calibration_progress.setVisible(False)
                
                # If session is active, determine posture quality.
                if self.session_active:
                    self.total_frames += 1
                    current_time = time.time()
                    if self.view_mode == "Front":
                        current_measurement = abs(nose_point[0] - shoulder_mid[0])
                        # Allow a 10% deviation from calibrated measurement.
                        tolerance_ratio = 1.1
                        is_good_posture = current_measurement <= self.calibrated_measurement * tolerance_ratio
                    else:
                        def angle(a, b, c):
                            a, b, c = np.array(a), np.array(b), np.array(c)
                            ab = a - b
                            bc = c - b
                            cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
                            return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
                        hip_mid = ((left_hip[0] + right_hip[0]) / 2,
                                   (left_hip[1] + right_hip[1]) / 2)
                        current_measurement = angle(hip_mid, shoulder_mid, nose_point)
                        # Allow a deviation within 5 degrees
                        tolerance_deg = 5
                        is_good_posture = abs(current_measurement - self.calibrated_measurement) <= tolerance_deg
                    
                    # Update session stats and posture status display
                    if is_good_posture:
                        self.good_posture_frames += 1
                        self.posture_status_label.setText("GOOD POSTURE")
                        self.posture_status_label.setStyleSheet("""
                            background-color: #2ecc71;
                            color: white;
                            font-size: 24px;
                            font-weight: bold;
                            font-family: 'Segoe UI', sans-serif;
                            padding: 10px;
                            border-radius: 5px;
                            margin: 10px;
                        """)
                    else:
                        self.poor_posture_frames += 1
                        self.posture_status_label.setText("POOR POSTURE - SIT UP STRAIGHT!")
                        self.posture_status_label.setStyleSheet("""
                            background-color: #e74c3c;
                            color: white;
                            font-size: 24px;
                            font-weight: bold;
                            font-family: 'Segoe UI', sans-serif;
                            padding: 10px;
                            border-radius: 5px;
                            margin: 10px;
                        """)
                        if current_time - self.last_alert_time > self.alert_cooldown:
                            self.speak_async("Warning! Poor posture detected.")
                            self.last_alert_time = current_time
                            self.poor_alerts_count += 1
                    
                    elapsed = int(current_time - self.session_start_time)
                    hrs, rem = divmod(elapsed, 3600)
                    mins, secs = divmod(rem, 60)
                    self.session_time_card.setText(f"📊 Session Time: {hrs:02}:{mins:02}:{secs:02}")
                    good_pct = (self.good_posture_frames / self.total_frames) * 100
                    self.good_posture_card.setText(f"✅ Good Posture %: {good_pct:.0f}%")
                    self.poor_posture_card.setText(f"❌ Poor Posture Alerts: {self.poor_alerts_count}")
                    
                    # (Optional) Compute a posture score:
                    if self.view_mode == "Front":
                        deviation = current_measurement - self.calibrated_measurement
                        score = max(0, 100 * (1 - (deviation / (self.calibrated_measurement * 0.1))))
                    else:
                        deviation = abs(current_measurement - self.calibrated_measurement)
                        score = max(0, 100 * (1 - (deviation / tolerance_deg)))
                    self.posture_score_bar.setValue(int(score))
                    
                    # For side view, update color gradient feedback based on score.
                    r = int(255 * (100 - score) / 100)
                    g = int(255 * score / 100)
                    self.feedback_bar.setStyleSheet(f"background-color: rgb({r},{g},0);")
                
                # Draw bounding box around the detected pose.
                xs = [int(lm.x * width) for lm in landmarks]
                ys = [int(lm.y * height) for lm in landmarks]
                if xs and ys:
                    cv2.rectangle(frame, (min(xs), min(ys)), (max(xs), max(ys)), (255, 0, 0), 2)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(img))
        except Exception as e:
            print(f"Error in frame processing: {e}")
    
    def start_calibration(self):
        """Begin calibration by resetting calibration measurements and showing progress."""
        self.calibrated = False
        self.calibration_frames = self.total_calibration_frames
        self.calibration_measurements = []
        self.speak_async("Starting calibration. Please sit in your ideal posture position.")
        self.calibration_progress.setValue(0)
        self.calibration_progress.setVisible(True)
    
    def start_session(self):
        """Start a new posture monitoring session."""
        if not self.calibrated:
            self.speak_async("Please calibrate first before starting a session.")
            return
        self.session_active = True
        self.session_start_time = time.time()
        self.total_frames = 0
        self.good_posture_frames = 0
        self.poor_posture_frames = 0
        self.poor_alerts_count = 0
        self.last_alert_time = 0
        self.start_btn.setEnabled(False)
        self.end_btn.setEnabled(True)
        self.speak_async("Session started. Maintaining good posture!")
    
    def end_session(self):
        """End session and log results."""
        if not self.session_active:
            return
        self.session_active = False
        duration = int(time.time() - self.session_start_time)
        if self.total_frames > 0:
            good_pct = (self.good_posture_frames / self.total_frames) * 100
            bad_pct = (self.poor_posture_frames / self.total_frames) * 100
            try:
                with open(SESSION_LOG_FILE, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        datetime.fromtimestamp(self.session_start_time).strftime('%Y-%m-%d %H:%M:%S'),
                        duration,
                        f"{good_pct:.1f}%",
                        f"{bad_pct:.1f}%"
                    ])
            except Exception as e:
                print(f"Error saving session log: {e}")
            feedback = (
                f"Session ended. You maintained good posture {good_pct:.1f}% of the time. "
                f"Poor posture was detected {bad_pct:.1f}% of the time."
            )
            self.speak_async(feedback)
        self.start_btn.setEnabled(True)
        self.end_btn.setEnabled(False)
    
    def log_posture_event(self, measurement, is_good):
        """Log a posture event to file."""
        try:
            with open(POSTURE_LOG_FILE, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    f"{measurement:.1f}",
                    "Good" if is_good else "Poor"
                ])
        except Exception as e:
            print(f"Error logging posture event: {e}")
    
    def load_logs(self):
        """Load logs and display in the log viewers."""
        try:
            with open(POSTURE_LOG_FILE, 'r') as f:
                reader = csv.reader(f)
                posture_log = "\n".join(["\t".join(row) for row in reader])
                self.posture_log_viewer.setPlainText(posture_log)
            with open(SESSION_LOG_FILE, 'r') as f:
                reader = csv.reader(f)
                session_log = "\n".join(["\t".join(row) for row in reader])
                self.session_log_viewer.setPlainText(session_log)
        except FileNotFoundError as e:
            self.posture_log_viewer.setPlainText(f"Error loading logs: {str(e)}")
        except Exception as e:
            print(f"Error loading logs: {e}")
    
    def export_logs(self):
        """Export logs to a file, using the default export folder if set."""
        options = QFileDialog.Options()
        current_tab = self.log_tabs.currentIndex()
        default_name = "posture_logs.csv" if current_tab == 0 else "session_logs.csv"
        initial_dir = self.export_folder if self.export_folder else ""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export Logs", os.path.join(initial_dir, default_name),
            "CSV Files (.csv);;All Files ()", options=options
        )
        if file_name:
            try:
                source_file = POSTURE_LOG_FILE if current_tab == 0 else SESSION_LOG_FILE
                with open(source_file, 'r') as source, open(file_name, 'w') as target:
                    target.write(source.read())
                QMessageBox.information(self, "Success", "Logs exported successfully")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to export logs: {str(e)}")
    
    def speak_async(self, text):
        """Speak the text asynchronously if voice is enabled."""
        if self.voice_enabled and hasattr(self, 'engine') and self.engine:
            threading.Thread(
                target=lambda: self.engine.say(text) or self.engine.runAndWait(),
                daemon=True
            ).start()
    
    def closeEvent(self, event):
        """Release resources on close."""
        self.cap.release()
        if hasattr(self, 'engine') and self.engine:
            self.engine.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    win = PostureApp()
    win.show()
    sys.exit(app.exec_())