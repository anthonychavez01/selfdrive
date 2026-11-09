import sys
import os
import time
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QFrame
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt

import model
from send_steering_angle import send_angle
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class AutoSteeringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚗 Autonomous Steering Dashboard")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("""
            QWidget {
                background-color: #121212;
                color: #EEEEEE;
                font-family: Segoe UI;
            }
            QPushButton {
                background-color: #1E88E5;
                border-radius: 10px;
                padding: 10px;
                color: white;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QLabel {
                font-size: 16px;
            }
        """)

        # --- Main UI Layout ---
        main_layout = QVBoxLayout()

        # Header
        header = QLabel("🚘 Autonomous Steering Control Dashboard")
        header.setFont(QFont("Segoe UI", 22, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #90CAF9; margin: 15px;")
        main_layout.addWidget(header)

        # Display Area
        display_layout = QHBoxLayout()
        self.video_label = QLabel("Video / Camera Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setFrameShape(QFrame.Box)
        self.video_label.setStyleSheet("border: 2px solid #42A5F5;")
        self.video_label.setFixedSize(700, 400)

        self.steering_label = QLabel("Steering Wheel")
        self.steering_label.setAlignment(Qt.AlignCenter)
        self.steering_label.setFrameShape(QFrame.Box)
        self.steering_label.setStyleSheet("border: 2px solid #42A5F5;")
        self.steering_label.setFixedSize(350, 350)

        display_layout.addWidget(self.video_label)
        display_layout.addWidget(self.steering_label)
        main_layout.addLayout(display_layout)

        # Status Label
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Segoe UI", 12))
        self.status_label.setStyleSheet("color: #BDBDBD; margin-top: 10px;")
        main_layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()
        self.btn_select_folder = QPushButton("📁 Select Folder")
        self.btn_select_video = QPushButton("🎞️ Select Video File")
        self.btn_camera = QPushButton("📷 Use Live Camera")
        self.btn_start = QPushButton("▶ Start Prediction")
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_quit = QPushButton("❌ Quit")

        for b in [self.btn_select_folder, self.btn_select_video, self.btn_camera, self.btn_start, self.btn_stop, self.btn_quit]:
            b.setFixedWidth(180)
            button_layout.addWidget(b)

        main_layout.addLayout(button_layout)

        # Central Widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Button Connections
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.btn_select_video.clicked.connect(self.select_video)
        self.btn_camera.clicked.connect(self.use_camera)
        self.btn_start.clicked.connect(self.start_prediction)
        self.btn_stop.clicked.connect(self.stop_prediction)
        self.btn_quit.clicked.connect(self.close)

        # Variables
        self.input_mode = None      # "folder", "video", "camera"
        self.dataset_path = None
        self.video_capture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_index = 0
        self.smoothed_angle = 0
        self.session = None

        # Load steering wheel image
        self.steering_img = cv2.imread(resource_path("steering_wheel_image.jpg"), 0)

        if self.steering_img is None:
            raise FileNotFoundError("steering_wheel_image.jpg not found.")
        self.rows, self.cols = self.steering_img.shape

    # ==================== INPUT MODES ==================== #
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.dataset_path = folder
            self.input_mode = "folder"
            self.status_label.setText(f"✅ Folder Selected: {folder}")

    def select_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.dataset_path = video_path
            self.input_mode = "video"
            self.status_label.setText(f"🎞️ Video Selected: {video_path}")

    def use_camera(self):
        self.input_mode = "camera"
        self.status_label.setText("📷 Using Live Camera Feed")

    # ==================== MODEL LOADING ==================== #
    def start_prediction(self):
        if not self.input_mode:
            self.status_label.setText("⚠️ Please select an input source first.")
            return

        self.status_label.setText("⏳ Loading model...")
        QApplication.processEvents()

        import importlib
        tf.reset_default_graph()
        importlib.reload(model)
        self.session = tf.InteractiveSession()
        _ = model.y
        saver = tf.train.Saver()
        saver.restore(self.session, "save/model.ckpt")

        self.status_label.setText("✅ Model loaded. Running predictions...")
        QApplication.processEvents()

        if self.input_mode == "folder":
            self.frame_index = 0
        elif self.input_mode == "video":
            self.video_capture = cv2.VideoCapture(self.dataset_path)
        elif self.input_mode == "camera":
            self.video_capture = cv2.VideoCapture(0)

        self.smoothed_angle = 0
        self.timer.start(20)

    def stop_prediction(self):
        self.timer.stop()
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        if self.session:
            self.session.close()
        self.status_label.setText("🛑 Prediction stopped.")

    # ==================== FRAME UPDATES ==================== #
    def update_frame(self):
        if self.input_mode == "folder":
            frame_path = os.path.join(self.dataset_path, f"{self.frame_index}.jpg")
            if not os.path.exists(frame_path):
                self.stop_prediction()
                self.status_label.setText("✅ Completed all frames.")
                return
            full_image = cv2.imread(frame_path)
            self.frame_index += 1

        elif self.input_mode in ["video", "camera"]:
            if self.video_capture is None:
                return
            ret, full_image = self.video_capture.read()
            if not ret:
                self.stop_prediction()
                self.status_label.setText("✅ Video Stream Ended.")
                return

        else:
            return

        if full_image is None:
            return

        # Predict steering angle
        image = cv2.resize(full_image[-150:], (200, 66)) / 255.0
        degrees = (
            self.session.run(model.y, feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0]
            * 180.0 / 3.14159265
        )

        send_angle(degrees)
        self.status_label.setText(f"🎯 Predicted Steering Angle: {degrees:.2f}°")

        # Display video
        rgb_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(700, 400, Qt.KeepAspectRatio))

        # Smooth + rotate steering
        if degrees != self.smoothed_angle:
            self.smoothed_angle += 0.2 * pow(abs((degrees - self.smoothed_angle)), 2.0 / 3.0) * (
                (degrees - self.smoothed_angle) / abs(degrees - self.smoothed_angle)
            )

        M = cv2.getRotationMatrix2D((self.cols / 2, self.rows / 2), -self.smoothed_angle, 1)
        dst = cv2.warpAffine(self.steering_img, M, (self.cols, self.rows))
        qt_steering = QImage(dst.data, self.cols, self.rows, self.cols, QImage.Format_Grayscale8)
        self.steering_label.setPixmap(QPixmap.fromImage(qt_steering).scaled(350, 350, Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.stop_prediction()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoSteeringApp()
    window.show()
    sys.exit(app.exec_())
