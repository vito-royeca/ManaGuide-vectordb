import cv2

from picamera2 import Picamera2, Preview

# import os
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)

picam2.start_preview(Preview.QTGL)
picam2.start()
image = picam2.capture_array()

image = cv2.resize(image, (320, 240))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

cv2.imshow('test', image)