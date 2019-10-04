import cv2 as cv
import os
import time
import click

@click.command()
@click.option('--model-name', default='haarcascade_frontalface_default.xml',
              help='The name of the pre-trained model to load. Download more from https://github.com/opencv/opencv/tree/master/data/haarcascades')
@click.option('--camera-id', default=1,
              help='The id of the camera to use.. You can discover the connected cameras by runnimg: ls -ltrh /dev/video*.')
@click.option('--trt-optimize', default=False,
              help='Setting this to True, the downloaded TF model will be converted to TensorRT model.', is_flag=True)
def detector(model_name, camera_id, trt_optimize):

    detector_model = f'./models/{model_name}'
    classifier = cv.CascadeClassifier()
    if not classifier.load(detector_model):
        raise ValueError(f'Could not find {detector_model}')

    video_capture = cv.VideoCapture(camera_id)
    start_time = time.time()

    while(True):
        # Capture frame-by-frame
        video_capture_result, frame = video_capture.read()

        if video_capture_result == False:
            raise ValueError(f'Error reading the frame from camera {camera_id}')

        # face detection and other logic goes here
        faces = classifier.detectMultiScale(frame, 1.3, 5)

        for (x, y, w, h) in faces:
            # send each face in mqtt topic
            cv.rectangle(frame, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)

        cv.putText(frame, f"FPS:{ 1.0 / (time.time() - start_time):0.1f}",
                (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        start_time = time.time()

        cv.imshow('Input', frame)
        if cv.waitKey(1) == 27:
            break


if __name__ == "__main__":
    detector()
