import sys
import svgwrite
import cv2
import numpy as np
# import imutils
import time
import argparse
import redis

import tensorflow as tf
import os
import sys
import pickle
from PIL import Image
import copy

from yolo_utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image, convert_to_original_size

from imutils.video import FileVideoStream


def draw_svg(boxes, frame, cls_names, detection_size, is_letter_box_image=True):
    boxes = copy.deepcopy(boxes)
    dim1, dim2 = frame.shape[1], frame.shape[0]
    svg_document = svgwrite.Drawing(size=(dim1, dim2))
    svg_document.add(svg_document.rect(insert = (0, 0),
                       size = (dim1, dim2),
                       stroke_width = "10",
                       stroke = "green"
                       ,fill = "rgb(0,0,0)", fill_opacity=0
                      ))

    for cls, bboxs in copy.deepcopy(boxes).items():
        name = cls_names[cls]
        if "traffic" in name: continue
        for box, score in bboxs:
            bb = convert_to_original_size(box, np.array(detection_size),
                                          (frame.shape[1], frame.shape[0]),
                                           is_letter_box_image)
            
            startX = bb[0]
            startY = bb[1]
            box_w = bb[2] - startX
            box_h = bb[3] - startY
            
            svg_document.add(svg_document.rect(insert=(int(startX), int(startY)),
                            size=("{}px".format(box_w), "{}px".format(box_h)),
                            stroke_width="5",
                            stroke="yellow",
                            fill="rgb(0,0,0)", fill_opacity=0)
                            )

    for cls, bboxs in boxes.copy().items():  
        name = cls_names[cls]
        if "traffic" in name: continue

        for box, score in bboxs:
            # TODO change to reuse code
            bb = convert_to_original_size(box, np.array(detection_size),
                                          (frame.shape[1], frame.shape[0]),
                                           is_letter_box_image)
            
            startX = bb[0]
            startY = bb[1]
            box_w = bb[2] - startX
            box_h = bb[3] - startY
            
            #text = "{} {:.2f}".format(name, score*100)
            text = name
            text_style = "font-size:15px; font-family:Courier New; stroke:yellow; stroke-width:0.2em;"
            svg_document.add(svg_document.text(text, insert=(int(startX),
                                                 int(startY)+2),
                                     fill="black", style=text_style))

            text_style = "font-size:15px; font-family:Courier New;"
            svg_document.add(svg_document.text(text, insert=(int(startX),
                                                 int(startY)+2),
                                     fill="black", style=text_style))

    return svg_document.tostring()

def yolo_write_svg(stream_url):

    print("[INFO] opening redis connection")
    redis_db = redis.StrictRedis(host="localhost", port=6379, db=0)
    # load our serialized model from disk
   
    print("[INFO] loading TF model")
    frozenGraph = load_graph("frozen_darknet_yolov3_model.pb")
    boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)
    conf_threshold = 0.5
    iou_threshold = 0.4
    size = 416
    classes = load_coco_names("coco.names")


    cap = cv2.VideoCapture(stream_url)
#    fvs = FileVideoStream(stream_url, queue_size=1).start()

    print("[INFO] Starting")
    with tf.Session(graph=frozenGraph) as sess:
        print("[INFO] Started graph")
        while(True): 
            if True:
                cap = cv2.VideoCapture(stream_url)
                #for i in range(10):
                #    cap.grab()
                ret, frame = cap.read()
                #ret, frame = cap.retrieve()
                if not ret:
                    continue
            else:
        #while fvs.more():
                if not fvs.more():
                    continue
                frame = fvs.read()
            

            print("[INFO] Grabbed frame")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = letter_box_image(Image.fromarray(frame), size, size, 128).astype(np.float32)
            
            detected_boxes = sess.run(
            boxes, feed_dict={inputs: [img_resized]})
            
            filtered_boxes = non_max_suppression(detected_boxes,
                                             confidence_threshold=conf_threshold,
                                             iou_threshold=iou_threshold)

            svg_string = draw_svg(filtered_boxes, frame, classes, (size, size), True)

            print("write")
            redis_db.set('overlay', svg_string)
            print("done")

if __name__ == '__main__':
    stream_url = "http://127.0.0.1:8090/pattern.webm"
    #stream_url = "rtsp://73.241.109.34:8554/unicast"
    #stream_url = "http://52.23.243.107:8090/pattern.webm"

    # write_svg_haar(stream_url)
    yolo_write_svg(stream_url) 
