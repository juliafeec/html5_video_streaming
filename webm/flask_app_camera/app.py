from flask import Flask, render_template, send_from_directory
import svgwrite
import cv2

app = Flask(__name__)

stream_url = "http://127.0.0.1:8090/pattern.webm"

@app.route('/svg')
def svg():
    capture = cv2.VideoCapture(stream_url)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    read_flag, img = capture.read()

    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minSize=(20, 20))
    svg_document = svgwrite.Drawing(size = ("1280px", "720px"))
    svg_document.add(svg_document.rect(insert = (0, 0),
                                       size = ("1280px", "720px"),
                                       stroke_width = "10",
                                       stroke = "green"
                                       ,fill = "rgb(0,0,0)", fill_opacity=0
                                      ))
    
    for (x, y, w, h) in faces:
        x = int(x)
        y = int(y)
        svg_document.add(svg_document.rect(insert = (x, y),
                                           size = ("{}px".format(w), "{}px".format(h)),
                                           stroke_width = "10",
                                           stroke = "yellow"
                                           ,fill = "rgb(0,0,0)", fill_opacity=0
                                          ))
    
    svg_string = "data:image/svg+xml;utf8,"+svg_document.tostring()
    return svg_string


@app.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('static', path)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
