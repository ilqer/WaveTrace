#!/usr/bin/env python3
"""Pi-side MJPEG camera uploader — streams the Pi Camera feed to the PC for ground-truth labeling.

Run ON THE RASPBERRY PI:   python3 camera_stream.py
The PC then reads it at:    http://<pi-ip>:8090/stream.mjpg

This is a TRAINING-ONLY component: the camera supervises the CSI labels (CameraLabeler on the PC).
The deployed detector needs no camera. Requires picamera2:  sudo apt install -y python3-picamera2
"""
import io
import socketserver
from http import server
from threading import Condition

from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

PORT = 8090
SIZE = (1280, 720)


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class Handler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/stream.mjpg":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()
        try:
            while True:
                with output.condition:
                    output.condition.wait()
                    frame = output.frame
                self.wfile.write(b"--FRAME\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # client (PC) disconnected — normal


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": SIZE}))
output = StreamingOutput()
picam2.start_recording(MJPEGEncoder(), FileOutput(output))
print(f"camera streaming on http://0.0.0.0:{PORT}/stream.mjpg")
try:
    StreamingServer(("", PORT), Handler).serve_forever()
finally:
    picam2.stop_recording()
