#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, Response
from flask import send_from_directory

from camera import Camera

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_image')
def video_image():
    """Video streaming generator function."""
    frame = Camera().get_frame()
    return send_file(
    io.BytesIO(frame),
    mimetype='image/jpeg',
    as_attachment=True,
    attachment_filename='img')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
