import numpy as np
import os
import threading
import time
import uuid
import cv2 as cv
import tensorflow.contrib.tensorrt as trt
import tensorflow as tf

# Initialize from ENV
if 'CAMERA_ID' in os.environ:
    camera_id = int(os.environ['CAMERA_ID'])
else:
    camera_id = 1

if 'DETECTOR_MODEL' in os.environ:
    detector_model = int(os.environ['DETECTOR_MODEL'])
else:
    detector_model = './src/tensorflow-detector/model/ssd_inception_v2_coco_trt.pb'
    
if not os.path.exists(detector_model):
    raise ValueError(f'Could not load {detector_model}')

video_capture = cv.VideoCapture(camera_id)

trt_graph = tf.compat.v1.GraphDef()
print(f'Loading model {detector_model}...')
with tf.io.gfile.GFile(detector_model, 'rb') as f:
    trt_graph.ParseFromString(f.read())
    print(f'{detector_model} loaded.')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

tf_sess = tf.Session(config=tf_config)

tf.import_graph_def(trt_graph, name='')

tf_input = tf_sess.graph.get_tensor_by_name('image_tensor:0')
tf_scores = tf_sess.graph.get_tensor_by_name('detection_scores:0')
tf_boxes = tf_sess.graph.get_tensor_by_name('detection_boxes:0')
tf_classes = tf_sess.graph.get_tensor_by_name('detection_classes:0')
tf_num_detections = tf_sess.graph.get_tensor_by_name('num_detections:0')

while(True):
    # Capture frame-by-frame
    video_capture_result, frame = video_capture.read()

    if video_capture_result == False:
        print(f'Error reading the frame from camera {camera_id}')

    # face detection and other logic goes here
    image_resized = cv.resize(frame,(300,300))

    scores, boxes, classes, num_detections = tf_sess.run(
        [tf_scores, tf_boxes, tf_classes, tf_num_detections], 
        feed_dict={tf_input: image_resized[None, ...]})

    boxes = boxes[0] # index by 0 to remove batch dimension
    scores = scores[0]
    classes = classes[0]
    num_detections = num_detections[0]
    
    print('----------------------------------------------------')
    for i in range(int(num_detections)):
        box = boxes[i] * np.array([300, 300, 300, 300])

        detected_object = image_resized[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
        detected_object = cv.resize(detected_object, (256,256), interpolation = cv.INTER_AREA) 
        ret, face_jpg = cv.imencode(".jpg", detected_object) 
        client.publish(f'device/{deviceId}/object/{int(classes[i])}', face_jpg.tobytes())
        print(f'Detected object class {int(classes[i])}')

client.loop_stop()
