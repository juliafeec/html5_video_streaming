import svgwrite
import cv2
import numpy as np
# import imutils
import time
import argparse
import redis

def write_svg_facenet(stream_url):
    print("[INFO] opening redis connection")
    redis_db = redis.StrictRedis(host="localhost", port=6379, db=0)
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    process_this_frame = True
    while(True):
        # time.sleep(0.1)
        if process_this_frame:
            capture = cv2.VideoCapture(stream_url)
            ret, frame = capture.read()
            # cv2.imshow('Stream IP camera opencv', frame)

            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            # frame = vs.read()
            # frame = imutils.resize(frame, width=400)

            # grab the frame dimensions and convert it to a blob
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                        (300, 300), (104.0, 177.0, 123.0))



            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()

            svg_document = svgwrite.Drawing(size=("1280px", "720px"))

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence < args["confidence"]:
                    continue

                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the bounding box of the face along with the associated
                # probability
                box_w = endX-startX
                box_h = endY-startY

                # draw the bounding box of the face along with the associated
                # probability
                svg_document.add(svg_document.rect(insert=(int(startX), int(startY)),
                                        size=("{}px".format(box_w), "{}px".format(box_h)),
                                        stroke_width="10",
                                        stroke="yellow",
                                        fill="rgb(0,0,0)", fill_opacity=0)
                                        )
                text = "{:.2f}%".format(confidence * 100)
                text_style = "font-size:%ipx; font-family:%s" % (20, "Courier New")
                svg_document.add(svg_document.text(text, insert=(int(startX),
                                                                int(startY)+20),
                                                    fill="black", style=text_style))
            # cv2.imshow("Frame", frame)
            # print(svg_document.tostring())
            # with open('overlay.xml', 'w') as f:
                # f.write(svg_document.tostring())
            redis_db.set('overlay', svg_document.tostring())
        
        process_this_frame = not process_this_frame
        
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def write_svg_haar(stream_url):
    print("[INFO] opening redis connection")
    redis_db = redis.StrictRedis(host="localhost", port=6379, db=0)
    print("[INFO] starting stream")
    capture = cv2.VideoCapture(stream_url)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 0)
    process_this_frame = True
    while True:
        if process_this_frame: 
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
            # print(svg_document.tostring())
            # with open('overlay.xml', 'w') as f:
                # f.write(svg_document.tostring())
            redis_db.set('overlay', svg_document.tostring())

        process_this_frame = not process_this_frame
       
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

if __name__ == '__main__':
    stream_url = "http://127.0.0.1:8090/pattern.webm"
    # write_svg_haar(stream_url)
    write_svg_facenet(stream_url) 