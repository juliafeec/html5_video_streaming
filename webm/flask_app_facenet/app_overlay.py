from flask import Flask, render_template, send_from_directory
import svgwrite
import cv2
import numpy as np
# import imutils
import time
import argparse
import base64
import redis

redis_db = redis.StrictRedis(host="localhost", port=6379, db=0)
app = Flask(__name__)

@app.route("/svg")
def svg():
    # with open('overlay.xml', 'r') as f:
        # svg_overlay = f.read()
    svg_overlay = redis_db.get('overlay').decode('utf-8')
    svg_string = "data:image/svg+xml;utf8,"+svg_overlay
    return svg_string
    # return render_template('index_stream.html', img=frame_b64)


@app.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('static', path)


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
