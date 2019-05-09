import sys
import os
sys.path.append(os.path.join(os.path.expanduser('~'), 'facenet', 'src'))
import svgwrite
import cv2
import numpy as np
# import imutils
import time
import argparse
import redis

from align import detect_face
from facenet import prewhiten, crop, flip, to_rgb

import tensorflow as tf
import facenet
import os
import sys
import pickle
from scipy import misc
from scipy.spatial.distance import cdist
from imutils.video import FileVideoStream
from glob import glob
import time

from boto.s3.key import Key
import boto
import boto3
from base64 import b64decode
from user_definition import key_id, access_key

def load_img(img, do_random_crop, do_random_flip, image_size,
             do_prewhiten=True):
    '''
    Process the captured images from the webcam, prewhitening, cropping and
    flipping as required. Returns processed image.
    '''
    images = np.zeros((1, image_size, image_size, 3))
    if img.ndim == 2:
        img = to_rgb(img)
    if do_prewhiten:
        img = prewhiten(img)
    img = crop(img, do_random_crop, image_size)
    img = flip(img, do_random_flip)
    images[:, :, :, :] = img
    return images

def load_and_align_data(image_paths, image_size=160, margin=44):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    img_list = []
    nrof_samples = len(image_paths)
    success_paths = []
    for img_path in image_paths:
        img = misc.imread(img_path, mode='RGB')
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if len(bounding_boxes) < 1:
            print("no face found in {}".format(img_path))
            continue
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        prewhitened = prewhiten(aligned)
        img_list.append(prewhitened)
        success_paths.append(img_path)
    images = np.stack(img_list)
    return success_paths, images

def resize_image(img, image_size, do_prewhiten=True):
    print(img.shape)
    if img.ndim == 2:
        img = to_rgb(img)
    if do_prewhiten:
        img = prewhiten(img)
    img = crop(img, False, image_size)
    img = flip(img, False)
    img = np.expand_dims(img, axis=0)
    #images[:,:,:,:] = img
    #return images
    return img

def align_face(img, pnet, rnet, onet):
    '''
    Detect and align faces from a frame, returning the detected faces and
    the bounding boxes for the faces.
    '''
    print("start detect")
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    image_size = 160

    if img.size == 0:
        print("empty array")
        return False, img, [0, 0, 0, 0]

    if img.ndim < 2:
        print('Unable to align')

    if img.ndim == 2:
        img = to_rgb(img)

    img = img[:, :, 0:3]
    margin = 44

    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    print("done detect")
    nrof_faces = bounding_boxes.shape[0]
    detect_multiple_faces = True
    
    if nrof_faces == 0:
        return False, img, [0, 0, 0, 0]
    else:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces > 1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:, 2]-det[:, 0])*(det[:, 3]-det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack([(det[:, 0]+det[:, 2])/2-img_center[1],
                                     (det[:, 1]+det[:, 3])/2-img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                # some extra weight on the centering
                index = np.argmax(bounding_box_size-offset_dist_squared*2.0)
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))
        if len(det_arr) > 0:
                faces = []
                bboxes = []
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            scaled = misc.imresize(cropped, (image_size, image_size), 
                                   interp='bilinear')
            faces.append(scaled)
            bboxes.append(bb)
        print("leaving align face")
        return True, faces, bboxes


def identify_person(image_vector, feature_names, feature_np, k=9):
    '''
    Calculates the Euclidean distance between a face embedding and the
    stored embeddings, returning the identity of the stored embedding most
    similar to the face embedding and the distance between these embeddings.
    '''
    d = np.squeeze(cdist(image_vector, feature_np, metric='euclidean'))
    top_k_ind = np.argsort(d).tolist()[:k]
    result = feature_names[top_k_ind[0]]
    distance = d[top_k_ind[0]]
    name = result.split("_")[0]
    return name, distance


def write_svg_facenet_emb(stream_url):
    '''
    Reads the facenet model and the saved embeddings from disk, and connects to
    the in-memory Redis database. Detects faces in the specified stream and
    calculates the corresponding bounding boxes. Writes the bounding boxes for
    all detected and identified faces to an svg overlay which is then saved to
    Redis to be accessed by other processes.
    '''
    dim1, dim2 = "1280px", "720px"

    print("[INFO] opening redis connection")
    redis_db = redis.StrictRedis(host="localhost", port=6379, db=0)
    # load our serialized model from disk
    print("[INFO] loading model...")
    #with open('extracted_dict.pickle', 'rb') as f:
    # TODO CHANGE THIS
    with open('startup_extracted_dict.pickle', 'rb') as f:
    #with open('byron.pickle', 'rb') as f:
        feature_dict = pickle.load(f)
    
    feature_names = np.array(list(feature_dict.keys()))
    feature_np = np.squeeze(list(feature_dict.values()))

    # model_exp = "20180408-102900/"
    model_exp = "20180402-114759"
    graph_fr = tf.Graph()
    sess_fr = tf.Session(graph=graph_fr)

    with graph_fr.as_default():
        print("Loading graph")
        saverf = tf.train.import_meta_graph(os.path.join(model_exp,
                                            'model-20180402-114759.meta'))
        saverf.restore(sess_fr, os.path.join(model_exp,
                                             'model-20180402-114759.ckpt-275'))
    
        pnet, rnet, onet = detect_face.create_mtcnn(sess_fr, None)
        sess = sess_fr
        images_placeholder = sess.graph.get_tensor_by_name("input:0")
        images_placeholder = tf.image.resize_images(images_placeholder,
                                                    (160, 160))
        embeddings = sess.graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = sess.graph.\
            get_tensor_by_name("phase_train:0")
    
        image_size = 160
        embedding_size = embeddings.get_shape()[1]
        print("Starting prediction")

        #fvs = FileVideoStream(stream_url, queue_size=1).start()
        use_fvs = False
        use_buffer = True

        if use_fvs:
            fvs = FileVideoStream(stream_url, queue_size=30).start()
        else:
            capture = cv2.VideoCapture(stream_url)
            print("got capture")

        # capture = cv2.VideoCapture(stream_url)
        while(True):
            print("capture frame")

            # for k in range(100):
            #    start_time = time.time()
            #    r = capture.grab()
            #    delta = time.time() - start_time
            #    if delta > 0.09:
            #        break
            # ret, frame = capture.retrieve()

            # capture = cv2.VideoCapture(stream_url)
            # ret, frame = capture.read()
            
            if redis_db.get('create_embs'):
                print("create temp svg")
                svg_document = svgwrite.Drawing(size=(dim1, dim2))
                svg_document.add(svg_document.rect(
                 insert=(0, 0),
                 size=(dim1, dim2),
                 stroke_width="10",
                 stroke="green",
                 fill="rgb(0,0,0)",
                 fill_opacity=0)
                 )

                #text_style = "font-size:40px; font-family:Courier New;\
                text_style = "font-size:70px; font-family:Courier New;\
                      fill:rgb(255,0,0); "
                svg_document.add(svg_document.text(
                 "Adding",
                 insert=(100, 200),
                 fill="black",
                 style=text_style)
                 )
                text_style = "font-size:70px; font-family:Courier New;\
                                      fill:rgb(255,0,0); "
                svg_document.add(svg_document.text(
                 "person",
                 insert=(100, 300),
                 fill="black",
                 style=text_style)
                 )

                svg_string = svg_document.tostring()
                print("write tmp overlay")
                redis_db.set('overlay', svg_string)

                print("create embs now")
                filenames = glob("photos/*")
                success_filenames, aligned_images = load_and_align_data(filenames)
                print("success filenames \n\n\n\n\n\n")
                print(success_filenames)
                
                #bucket_name = 'msds603camera' # Change it to your bucket.
                #s3_connection = boto.connect_s3(aws_access_key_id=key_id,
                #                                aws_secret_access_key=access_key)
                #bucket = s3_connection.get_bucket(bucket_name)
                for fname, img in zip(success_filenames, aligned_images):
                    print("start run emb")
                    images = resize_image(img, image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                    name, count = fname.split("/")[-1].rsplit(".")[0].split("_")
                    feature_dict["{}_{}".format(name, count)] =  feature_vector
                    print("removing photo")

                    #k = Key(bucket)
                    #k.key = fname
                    #k.set_contents_from_filename(fname)
                    #key = bucket.lookup(fname)
                    #key.set_acl('public-read-write')
                    os.remove(fname)

                print("writing")
                
                with open('extracted_dict.pickle','wb') as f:
                    pickle.dump(feature_dict, f)
                    
                print("done writing")
                
                    #feed_dict = {images_placeholder: images,
                    #                 phase_train_placeholder: False}
                    #feature_vector = sess.run(embeddings, feed_dict=feed_dict)


                feature_names = np.array(list(feature_dict.keys()))
                print(feature_names)
                feature_np = np.squeeze(list(feature_dict.values()))

                redis_db.set('create_embs', '')
                    
                print("done creating embs")
                if use_buffer:
                    print("reopening capture")
                    capture = cv2.VideoCapture(stream_url)


            try: 
                if use_fvs:
                    print("capture frame with fvs")
                    if not fvs.more():
                        continue
                    frame = fvs.read()
                elif use_buffer:
                    print("capture frame with cv2 buffer")
                    time.sleep(1)
                    #time.sleep(2.5)
                    for i in range(50):
                        capture.grab()

                    ret, frame = capture.read()
                else:
                    print("capture frame with cv2")
                    capture = cv2.VideoCapture(stream_url)
                    ret, frame = capture.read()

                gray = cv2.cvtColor(frame, 0)
            except: 
                #continue
                raise Exception("capture failed")

            print("converted to gray")
            if gray.size < 0:
                print("skipping")
                continue

            print(gray.size)
            response, faces, bboxs = align_face(gray, pnet, rnet, onet)
            print(response)
            print("{} faces found.".format(len(faces)))
            svg_document = svgwrite.Drawing(size=(dim1, dim2))
            svg_document.add(svg_document.rect(
                                 insert=(0, 0),
                                 size=(dim1, dim2),
                                 stroke_width="10",
                                 #stroke_width="7",
                                 stroke="green",
                                 fill="rgb(0,0,0)",
                                 fill_opacity=0)
                                 )

            if response is True:
                #svg_document = svgwrite.Drawing(size=(dim1, dim2))
                #svg_document.add(svg_document.rect(
                #                 insert=(0, 0),
                #                 size=(dim1, dim2),
                #                 stroke_width="10",
                #                 #stroke_width="7",
                #                 stroke="green",
                #                 fill="rgb(0,0,0)",
                #                 fill_opacity=0)
                #                 )
                for i, image in enumerate(faces):
                    # 640 360
                    dim1, dim2 = frame.shape[1], frame.shape[0]
                    
                    bb = bboxs[i]
                    images = load_img(image, False, False, image_size)
                    feed_dict = {images_placeholder: images,
                                 phase_train_placeholder: False}
                    print("start run emb")
                    feature_vector = sess.run(embeddings, feed_dict=feed_dict)
                    print("start identify")
                    result, distance = identify_person(feature_vector,
                                                       feature_names,
                                                       feature_np,
                                                       1)
                    print("identified: %s, distance: %.3f" % (result,
                                                              distance))
                    print("calculate svg")
                    #if distance < 1.0:

                    if (1-distance) > 0.15:
                        print("name: {} distance: {} text: {}".format(result, distance, 1-distance))
                        startX = bb[0]
                        startY = bb[1]
                        box_w = bb[2] - startX
                        box_h = bb[3] - startY
                        
                        svg_document.add(svg_document.rect(
                                         insert=(int(startX), int(startY)),
                                         size=("{}px".format(box_w),
                                               "{}px".format(box_h)),
                                         #stroke_width="10",
                                         stroke_width="7",
                                         stroke="yellow",
                                         fill="rgb(0,0,0)",
                                         fill_opacity=0)
                                         )
                        text = "{} {:.2f}".format(result, 1-distance)
                        #text_style = "font-size:30px; font-family:Courier New;\
                        text_style = "font-size:50px; font-family:Courier New;\
                                      stroke:yellow; stroke-width:0.2em;"
                        svg_document.add(svg_document.text(
                                         text,
                                         insert=(int(startX), int(startY)+20),
                                         fill="black",
                                         style=text_style)
                                         )
                        
                        #text_style = "font-size:30px; font-family:Courier New;"
                        text_style = "font-size:50px; font-family:Courier New;"
                        svg_document.add(svg_document.text(
                                         text,
                                         insert=(int(startX), int(startY)+20),
                                         fill="black", 
                                         style=text_style)
                                         )
                    #elif (1-distance) > 0.05:
                    else:
                        print("name: {} distance: {}".format(result, distance))
                        startX = bb[0]
                        startY = bb[1]
                        box_w = bb[2] - startX
                        box_h = bb[3] - startY

                        svg_document.add(svg_document.rect(
                                         insert=(int(startX), int(startY)),
                                         size=("{}px".format(box_w),
                                               "{}px".format(box_h)),
                                         stroke_width="10",
                                         stroke="yellow",
                                         fill="rgb(0,0,0)",
                                         fill_opacity=0)
                                         )


               # print("export svg")
               # svg_string = svg_document.tostring()
               # print("write")
               # redis_db.set('overlay', svg_string)
               # print("done")
                
            print("export svg")
            svg_string = svg_document.tostring()
            print("write")
            redis_db.set('overlay', svg_string)
            print("done")



def write_svg_facenet(stream_url):
    '''
    Reads an alternative facenet model, and connects to the in-memory Redis 
    database. Detects faces (no identification) in the specified stream and 
    calculates the corresponding bounding boxes. Writes the bounding boxes for
    all detected faces to an svg overlay which is then saved to
    Redis to be accessed by other processes. 
    '''
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
                # extract the confidence (i.e., probability) associated with 
                # the prediction
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
                svg_document.add(svg_document.rect(
                                 insert=(int(startX), int(startY)),
                                 size=("{}px".format(box_w),
                                       "{}px".format(box_h)),
                                 stroke_width="10",
                                 stroke="yellow",
                                 fill="rgb(0,0,0)",
                                 fill_opacity=0)
                                 )
                text = "{:.2f}%".format(confidence * 100)
                text_style = "font-size:%ipx; \
                              font-family:%s" % (20, "Courier New")
                svg_document.add(svg_document.text(
                                 text,
                                 insert=(int(startX), int(startY)+20),
                                 fill="black",
                                 style=text_style)
                                 )
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


if __name__ == '__main__':
#    stream_url = "http://127.0.0.1:8090/pattern.webm"
    stream_url = "http://ec2-13-57-20-231.us-west-1.compute.amazonaws.com:8090/pattern.webm"
    # stream_url = "http://52.23.243.107:8090/pattern.webm"
    # write_svg_haar(stream_url)
    write_svg_facenet_emb(stream_url)

