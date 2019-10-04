import time
import numpy as np
import os
import cv2 as cv
import click
import time

from multithreading.videocaptureasync import VideoCaptureAsync
from multithreading.objectdetectasync import ObjectDetectionAsync

import threading


@click.command()
@click.option('--model-name', default='ssd_inception_v2_coco',
              help='The name of the pre-trained TF model to load. See: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md')
@click.option('--camera-id', default=1,
              help='The id of the camera to use.. You can discover the connected cameras by runnimg: ls -ltrh /dev/video*.')
@click.option('--trt-optimize', default=False,
              help='Setting this to True, the downloaded TF model will be converted to TensorRT model.', is_flag=True)
def detector(model_name, camera_id, trt_optimize):

    read_lock = threading.Lock()

    video_capture = VideoCaptureAsync(camera_id)
    video_capture.start()

    video_capture_result, frame, frame_resized = video_capture.read()
    if video_capture_result == False:
        raise ValueError(f'Error reading the frame from camera {camera_id}')
    camera_height, camera_width, channels = frame.shape

    detector_frame = None

    def frame_callback():
        with read_lock:
            if detector_frame is not None and len(detector_frame) > 0:
                return detector_frame.copy()
            return None

    object_detection = ObjectDetectionAsync(
        model_name, trt_optimize, frame_callback)
    object_detection.start()
    start_time = time.time()

    while(video_capture_result):
        # Capture frame-by-frame
        video_capture_result, frame, frame_resized = video_capture.read()
        with read_lock:
            detector_frame = frame_resized

        if video_capture_result == False:
            print(f'Error reading the frame from camera {camera_id}')

        if object_detection.ready:
            scores, boxes, classes, num_detections = object_detection.read()

            for i in range(int(num_detections)):
                box = boxes[i] * np.array([camera_height,
                                           camera_width, camera_height, camera_width])
                box = box.astype(int)

                cv.rectangle(frame, (box[1], box[0]), (box[3],
                                                       box[2]), color=(0, 255, 0), thickness=1)
                text = f"{scores[i]*100:.0f} | {str(int(classes[i]))}"
                cv.putText(
                    frame, text, (box[3]+10, box[2]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv.putText(frame, f"FPS:{ 1.0 / (time.time() - start_time):0.1f}",
                           (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                start_time = time.time()
        else:
            cv.putText(frame, f"Loading detector"+int(time.time() % 4)*".",
                       (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            time.sleep(0.1)

        cv.imshow('Input', frame)
        if cv.waitKey(1) == 27:
            break

    cv.destroyAllWindows()
    video_capture.stop()
    object_detection.stop()


if __name__ == "__main__":
    detector()
