# newapp2.py
import sys
import os
import re
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


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def numeric_sort_key(path):
    name = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)", name)
    if m:
        return int(m.group(1))
    try:
        return int(name)
    except Exception:
        return name.lower()


class AutoSteeringApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Autonomous Driving System")
        self.setGeometry(100, 80, 1280, 820)

        # ==== CONFIGURE YOUR DEFAULT PATHS (raw strings to avoid unicodeescape) ====
        # Default dataset folder for Auto Stream
        self.default_dataset_path = r"C:\Users\krish\Desktop\selfdrive\driving_dataset"
        # Default checkpoint path for model restore
        self.default_model_checkpoint = r"C:\Users\krish\Desktop\selfdrive\save\model.ckpt"
        # ==========================================================================

        # Dark modern theme
        self.setStyleSheet("""
            QMainWindow { background-color: #0f1417; }
            QWidget { background-color: transparent; color: #d6e6ef; }
            QLabel { color: #cfeaf8; }
            QFrame.card {
                background: #11161a;
                border: 1px solid rgba(255,255,255,0.05);
                border-radius: 12px;
            }
            QPushButton { border-radius: 8px; padding: 8px 14px; font-weight: 600; }
            QPushButton.primary { background-color: #27ae60; color: #fff; }
            QPushButton.danger { background-color: #c0392b; color: #fff; }
            QPushButton.neutral { background-color: #24323a; color: #d0e9f8; }
        """)

        # --- UI (no Accuracy card) ---
        main_layout = QVBoxLayout()
        main_layout.setSpacing(18)
        main_layout.setContentsMargins(28, 18, 28, 18)

        header_layout = QHBoxLayout()
        title_block = QVBoxLayout()
        title_label = QLabel("Autonomous Driving System")
        title_label.setFont(QFont("Segoe UI", 28, QFont.Bold))
        subtitle = QLabel("Real-time steering prediction and control")
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setStyleSheet("color: #7f9aa3;")
        title_block.addWidget(title_label)
        title_block.addWidget(subtitle)
        header_layout.addLayout(title_block)
        header_layout.addStretch()

        # Start / Stop
        self.btn_start_sim = QPushButton("Start Simulation")
        self.btn_start_sim.setProperty("class", "primary")
        self.btn_stop_sim = QPushButton("Stop")
        self.btn_stop_sim.setProperty("class", "danger")
        self.btn_start_sim.setFixedWidth(160)
        self.btn_stop_sim.setFixedWidth(120)
        header_layout.addWidget(self.btn_start_sim)
        header_layout.addWidget(self.btn_stop_sim)
        main_layout.addLayout(header_layout)

        # Metrics row
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(18)

        steering_card = QFrame()
        steering_card.setProperty("class", "card")
        s_layout = QVBoxLayout()
        s_title = QLabel("Steering Angle")
        s_title.setFont(QFont("Segoe UI", 12))
        self.steering_angle_label = QLabel("0.00°")
        self.steering_angle_label.setFont(QFont("Segoe UI", 26, QFont.Bold))
        self.steering_angle_label.setStyleSheet("color: #63c3ff;")
        s_layout.addWidget(s_title)
        s_layout.addStretch()
        s_layout.addWidget(self.steering_angle_label)
        steering_card.setLayout(s_layout)
        steering_card.setFixedWidth(300)

        fps_card = QFrame()
        fps_card.setProperty("class", "card")
        f_layout = QVBoxLayout()
        f_title = QLabel("Frame Rate")
        f_title.setFont(QFont("Segoe UI", 12))
        self.fps_label = QLabel("0 FPS")
        self.fps_label.setFont(QFont("Segoe UI", 26, QFont.Bold))
        self.fps_label.setStyleSheet("color: #66d37a;")
        f_layout.addWidget(f_title)
        f_layout.addStretch()
        f_layout.addWidget(self.fps_label)
        fps_card.setLayout(f_layout)
        fps_card.setFixedWidth(300)

        metrics_layout.addWidget(steering_card)
        metrics_layout.addWidget(fps_card)
        metrics_layout.addStretch()
        main_layout.addLayout(metrics_layout)

        # Bottom: steering wheel and feed
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(24)

        # Steering panel
        steering_panel = QFrame()
        steering_panel.setProperty("class", "card")
        st_layout = QVBoxLayout()
        st_layout.setContentsMargins(24, 24, 24, 24)
        st_title = QLabel("Current Angle")
        st_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.current_angle_label = QLabel("0.00°")
        self.current_angle_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.current_angle_label.setStyleSheet("color: #63c3ff;")
        self.current_angle_label.setAlignment(Qt.AlignCenter)
        self.steering_label = QLabel()
        self.steering_label.setFixedSize(420, 420)
        self.steering_label.setAlignment(Qt.AlignCenter)
        st_layout.addWidget(st_title, alignment=Qt.AlignHCenter)
        st_layout.addWidget(self.current_angle_label, alignment=Qt.AlignHCenter)
        st_layout.addWidget(self.steering_label, alignment=Qt.AlignHCenter)
        steering_panel.setLayout(st_layout)
        steering_panel.setFixedWidth(520)

        # Video panel
        video_panel = QFrame()
        video_panel.setProperty("class", "card")
        vid_layout = QVBoxLayout()
        vid_layout.setContentsMargins(20, 22, 20, 20)
        v_title = QLabel("Live Road Feed")
        v_title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        vid_layout.addWidget(v_title)

        self.btn_auto_stream = QPushButton("Auto Stream Dataset")
        self.btn_auto_stream.setProperty("class", "primary")
        self.btn_stop_stream = QPushButton("Stop")
        self.btn_stop_stream.setProperty("class", "danger")
        self.btn_choose_folder = QPushButton("Choose Folder")
        self.btn_choose_folder.setProperty("class", "neutral")
        b_row = QHBoxLayout()
        b_row.addWidget(self.btn_auto_stream)
        b_row.addWidget(self.btn_stop_stream)
        b_row.addWidget(self.btn_choose_folder)
        b_row.addStretch()
        vid_layout.addLayout(b_row)

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 360)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color:#071017; border-radius:10px;")
        vid_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        video_panel.setLayout(vid_layout)

        bottom_layout.addWidget(steering_panel)
        bottom_layout.addWidget(video_panel)
        main_layout.addLayout(bottom_layout)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # Buttons
        self.btn_start_sim.clicked.connect(self.start_prediction)
        self.btn_stop_sim.clicked.connect(self.stop_prediction)
        self.btn_auto_stream.clicked.connect(self.auto_stream_clicked)
        self.btn_stop_stream.clicked.connect(self.stop_prediction)
        self.btn_choose_folder.clicked.connect(self.choose_folder_clicked)

        # Variables & timers
        self.input_mode = None  # "folder"
        self.dataset_path = None
        self.image_files = []
        self.frame_index = 0
        self.timer = QTimer()
        self.timer.setInterval(25)  # 50 ms -> 20 FPS
        self.timer.timeout.connect(self.update_frame)
        self.fps_counter = 0
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(1000)
        self.smoothed_angle = 0.0
        self.session = None
        self.model_loaded = False
        self.video_capture = None
        self.last_selected_folder = None

        # Steering wheel image
        self.steering_img = cv2.imread(resource_path("steering_wheel_image.jpg"), cv2.IMREAD_GRAYSCALE)
        if self.steering_img is None:
            sz = 400
            placeholder = np.zeros((sz, sz), dtype=np.uint8)
            cv2.circle(placeholder, (sz//2, sz//2), sz//3, (200,), thickness=20)
            self.steering_img = placeholder
        self.rows, self.cols = self.steering_img.shape

        # Try to load model once immediately (but continue if it fails)
        self._try_load_model()

    def _try_load_model(self):
        """Attempt to load TensorFlow model once; on failure we continue without it."""
        # If already loaded, skip
        if self.model_loaded:
            return
        # Try multiple checkpoint locations: explicit default, relative 'save/model.ckpt'
        candidates = [self.default_model_checkpoint, os.path.join(resource_path("save"), "model.ckpt"), "save/model.ckpt"]
        for ckpt in candidates:
            try:
                if not os.path.exists(ckpt) and not os.path.exists(ckpt + ".index"):
                    # Not found - continue trying others
                    continue
                # Reset graph and create session
                tf.reset_default_graph()
                import importlib
                importlib.reload(model)
                self.session = tf.InteractiveSession()
                saver = tf.train.Saver()
                saver.restore(self.session, ckpt)
                self.model_loaded = True
                print(f"Model restored from: {ckpt}")
                return
            except Exception as e:
                print(f"Model load attempt failed for {ckpt}: {e}")
                # cleanup and continue
                try:
                    if self.session:
                        self.session.close()
                except Exception:
                    pass
                self.session = None
        print("Model not loaded; continuing without model (predictions will be 0.0).")
        self.model_loaded = False
        self.session = None

    def update_fps(self):
        self.fps_label.setText(f"{self.fps_counter} FPS")
        self.fps_counter = 0

    def choose_folder_clicked(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.dataset_path = folder
            self.last_selected_folder = folder
            self.input_mode = "folder"
            self.prepare_image_list()
            if not self.timer.isActive():
                self.timer.start()

    def auto_stream_clicked(self):
        if self.last_selected_folder and os.path.isdir(self.last_selected_folder):
            self.dataset_path = self.last_selected_folder
            self.input_mode = "folder"
        elif os.path.isdir(self.default_dataset_path):
            self.dataset_path = self.default_dataset_path
            self.input_mode = "folder"
        else:
            folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder (default not found)")
            if not folder:
                return
            self.dataset_path = folder
            self.last_selected_folder = folder
            self.input_mode = "folder"

        self.prepare_image_list()
        if not self.timer.isActive():
            self.timer.start()

    def prepare_image_list(self):
        if not self.dataset_path or not os.path.isdir(self.dataset_path):
            self.image_files = []
            return
        entries = [
            os.path.join(self.dataset_path, f)
            for f in os.listdir(self.dataset_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not entries:
            self.image_files = []
            return
        try:
            self.image_files = sorted(entries, key=numeric_sort_key)
        except Exception:
            self.image_files = sorted(entries)
        self.frame_index = 0

    def start_prediction(self):
        # ensure images prepared in folder mode
        if self.input_mode == "folder" and self.dataset_path:
            self.prepare_image_list()
            if not self.image_files:
                print("No images found in selected folder.")
                return
        # attempt to load model if not already loaded
        if not self.model_loaded:
            self._try_load_model()
        if not self.timer.isActive():
            self.timer.start()

    def stop_prediction(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.session:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None
            self.model_loaded = False
        if self.video_capture:
            try:
                self.video_capture.release()
            except Exception:
                pass
            self.video_capture = None

    def update_frame(self):
        if self.input_mode != "folder":
            return
        if not self.image_files:
            return

        img_path = self.image_files[self.frame_index % len(self.image_files)]
        self.frame_index += 1

        frame = cv2.imread(img_path)
        if frame is None:
            return

        # Predict steering angle
        degrees = 0.0
        try:
            h = frame.shape[0]
            crop = frame[h-150:h, :, :] if h >= 150 else frame
            image = cv2.resize(crop, (200, 66)).astype(np.float32) / 255.0
            if self.session is not None:
                # model.x, model.y, model.keep_prob expected in model.py
                degrees = (self.session.run(model.y, feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0]
                           * 180.0 / 3.14159265)
            else:
                degrees = 0.0
        except Exception as e:
            print("Prediction error (continuing):", e)
            degrees = 0.0

        # Send angle (safe)
        try:
            send_angle(degrees)
        except Exception as e:
            print("send_angle error (ignored):", e)

        # Update labels
        self.steering_angle_label.setText(f"{degrees:.2f}°")
        self.current_angle_label.setText(f"{degrees:.2f}°")

        # Display frame
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h_img, w_img, ch = rgb.shape
            bytes_per_line = ch * w_img
            qt_image = QImage(rgb.data, w_img, h_img, bytes_per_line, QImage.Format_RGB888)
            pix = QPixmap.fromImage(qt_image).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(pix)
        except Exception as e:
            print("Display error:", e)

        # Steering wheel rotation smoothing
        diff = degrees - self.smoothed_angle
        if abs(diff) > 1e-6:
            sign = 1 if diff > 0 else -1
            self.smoothed_angle += 0.2 * pow(abs(diff), 2.0 / 3.0) * sign

        try:
            M = cv2.getRotationMatrix2D((self.cols / 2, self.rows / 2), -self.smoothed_angle, 1)
            dst = cv2.warpAffine(self.steering_img, M, (self.cols, self.rows), borderMode=cv2.BORDER_REPLICATE)
            qt_steer = QImage(dst.data, self.cols, self.rows, self.cols, QImage.Format_Grayscale8)
            pix2 = QPixmap.fromImage(qt_steer).scaled(self.steering_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.steering_label.setPixmap(pix2)
        except Exception as e:
            print("Steering render error:", e)

        self.fps_counter += 1

    def closeEvent(self, event):
        # cleanup
        self.stop_prediction()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AutoSteeringApp()
    window.show()
    sys.exit(app.exec_())
