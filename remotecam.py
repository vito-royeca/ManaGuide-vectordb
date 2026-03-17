import socket
from threading import Event
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import PyavOutput

picam2 = Picamera2()
video_config = picam2.create_video_configuration({"size": (1280, 720), 'format': 'YUV420'})
picam2.configure(video_config)
encoder = H264Encoder(bitrate=10000000)
#encoder.audio = True # enable audio

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 8888))
    
    while True:
        sock.listen() # wait for connection
        conn, addr = sock.accept() # accept connection
        output = PyavOutput(f"pipe:{conn.fileno()}", format="mpegts")
        event = Event()
        output.error_callback = lambda e: event.set() # notify of disconnection
        picam2.start_recording(encoder, output)
        event.wait() # wait for disconnection
        picam2.stop_recording()