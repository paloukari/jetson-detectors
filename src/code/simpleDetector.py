import numpy as np
import cv2 as cv
import os
import threading
import time
import uuid

if 'CAMERA_ID' in os.environ:
    camera_id = int(os.environ['CAMERA_ID'])
else:
    camera_id = 0
    

classifier_xml = './src/model/haarcascade_frontalface_default.xml'
classifier = cv.CascadeClassifier()

if not classifier.load(classifier_xml):
    print(f'Could not load {classifier_xml}')

video_capture = cv.VideoCapture(camera_id)


while(True):
    # Capture frame-by-frame
    video_capture_result, frame = video_capture.read()
    
    if video_capture_result == False:
        print(f'Error reading the frame from camera {camera_id}')

    # We don't use the color information, so might as well save space
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    
    # face detection and other logic goes here
    faces = classifier.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # send each face in mqtt topic
        print(f'x:{x}, y:{y}, w:{w}, h:{h}')
        face = gray[y:y+h, x:x+w]
        face = cv.resize(face, (256,256), interpolation = cv.INTER_AREA)  
        ret, face_jpg = cv.imencode(".jpg", face)   
    
    cv.imshow('Input', frame)
    if cv.waitKey(1) == 27:
        break     

client.loop_stop()