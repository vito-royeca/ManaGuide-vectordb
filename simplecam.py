from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)


picam2.start_preview(Preview.QTGL)
picam2.start()

metadata = picam2.capture_metadata()
controls = {c: metadata[c] for c in ["ExposureTime", "AnalogueGain", "ColourGains"]}
print(controls)
picam2.set_controls(controls)

time.sleep(5)

picam2.capture_file("test.jpg")




