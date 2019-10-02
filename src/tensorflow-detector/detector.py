import numpy as np
import os
import time
import uuid
import cv2 as cv
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import threading

from videocaptureasync import VideoCaptureAsync
from objectdetectasync import ObjectDetectionAsync

if 'CAMERA_ID' in os.environ:
    camera_id = int(os.environ['CAMERA_ID'])
else:
    camera_id = 1

if 'DETECTOR_MODEL' in os.environ:
    detector_model = int(os.environ['DETECTOR_MODEL'])
else:
    detector_model = './model/ssd_inception_v2_coco_trt.pb'

if not os.path.exists(detector_model):
    raise ValueError(f'Could not load {detector_model}')

#video_capture = cv.VideoCapture(camera_id)
video_capture = VideoCaptureAsync(camera_id)
video_capture.start()
video_capture_result, frame = video_capture.test()

if video_capture_result == False:
    raise ValueError(f'Error reading the frame from camera {camera_id}')

trt_graph = tf.GraphDef()

print(f'Loading model {detector_model}... (this might take some minutes)')


with tf.gfile.GFile(detector_model, 'rb') as f:
    trt_graph.ParseFromString(f.read())
    print(f'{detector_model} loaded.')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

tf_sess = tf.Session(config=tf_config)

print(f'Importing Graph..')
tf.import_graph_def(trt_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name('image_tensor:0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

camera_height, camera_width, channels = frame.shape
read_lock = threading.Lock()

detector_frame = None


def frame_callback():
    with read_lock:
        if detector_frame is not None:
            return detector_frame.copy()
        return None


object_detection = ObjectDetectionAsync(
    tf_sess, tf_input, tf_scores, tf_boxes, tf_classes, tf_num_detections, frame_callback)
object_detection.start()

while(video_capture_result):
    # Capture frame-by-frame
    video_capture_result, frame, frame_resized = video_capture.read()
    with read_lock:
        detector_frame = frame_resized

    if video_capture_result == False:
        print(f'Error reading the frame from camera {camera_id}')

    scores, boxes, classes, num_detections = object_detection.read()
    for i in range(int(num_detections)):
        box = boxes[i] * np.array([camera_height,
                                   camera_width, camera_height, camera_width])
        box = box.astype(int)

        cv.rectangle(frame, (box[1], box[0]), (box[3],
                                               box[2]), color=(0, 255, 0), thickness=1)
        text = "{0:.0f}% | {1}".format(scores[i]*100, str(int(classes[i])))
        cv.putText(frame,text,(box[3]+10, box[2]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv.imshow('Input', frame)
    if cv.waitKey(1) == 27:
        break

cv.destroyAllWindows()
video_capture.stop()
object_detection.stop()