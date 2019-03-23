import cv2
from base_camera import BaseCamera
import time

class Camera(BaseCamera):
    #video_source = 0
    video_source = 'http://127.0.0.1:8080/video_feed'

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Camera is not available.')

        while True:
            # read current frame
            ret, img = camera.read()
            if img is None:
                print("none")
                continue

            # add overlay
            overlay = img.copy()
            output = img
            
            alpha = 0.5
            cv2.rectangle(overlay, (200, 200), (300, 400),
            	(0, 122, 122), -1)
            cv2.putText(overlay, "TEST".format(alpha),
            	(30, 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 122, 122), 3)
            
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
            img = output

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
