import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QApplication, QWidget
from picamera2.previews.qt import QGlPicamera2
from picamera2 import Picamera2, Preview
import datetime
import time

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)

def on_button_clicked():
    button.setEnabled(False)
    
    current_time = datetime.datetime.now()
    # Format: YYYY-MM-DD_HH-MM-SS
    timestamp_string = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"Photo_{timestamp_string}.jpg"

    cfg = picam2.create_still_configuration()
    picam2.switch_mode_and_capture_file(cfg, filename, signal_function=qpicamera2.signal_done)

def capture_done(job):
    result = picam2.wait(job)
    button.setEnabled(True)

app = QApplication([])
qpicamera2 = QGlPicamera2(picam2, width=800, height=600, keep_ar=False)
button = QPushButton("Click to capture JPEG")
window = QWidget()
qpicamera2.done_signal.connect(capture_done)
button.clicked.connect(on_button_clicked)

layout_v = QVBoxLayout()
layout_v.addWidget(qpicamera2)
layout_v.addWidget(button)
window.setWindowTitle("Qt Picamera2 App")
window.resize(640, 480)
window.setLayout(layout_v)

picam2.start()
window.show()
app.exec()

# time.sleep(5)
