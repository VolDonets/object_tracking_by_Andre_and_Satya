import dlib
import glob
import cv2
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import pyautogui as pyg
import shutil
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils

file_name = 'models/Hand_Detector_v8_c8.svm'
detector = dlib.simple_object_detector(file_name)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)

scale_factor = 2.0
size, center_x = 0, 0
fps = 0
frame_counter = 0
start_time = time.time()

is_not_detected_before = True
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}
tracker = OPENCV_OBJECT_TRACKERS["kcf"]()
initBB = None

while (True):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_counter += 1
    fps = (frame_counter / (time.time() - start_time))
    copy = frame.copy()
    new_width = int(frame.shape[1] / scale_factor)
    new_height = int(frame.shape[0] / scale_factor)
    resized_frame = cv2.resize(copy, (new_width, new_height))
    if is_not_detected_before:
        detections = detector(resized_frame)
        for detection in (detections):
            x1 = int(detection.left() * scale_factor)
            y1 = int(detection.top() * scale_factor)
            x2 = int(detection.right() * scale_factor)
            y2 = int(detection.bottom() * scale_factor)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 'Hand DETECTED', (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
            size = int((x2 - x1) * (y2 - y1))
            center_x = x2 - x1 // 2
            initBB = (detection.left(), detection.top(), detection.right() - detection.left(), detection.bottom() - detection.top())
            tracker = OPENCV_OBJECT_TRACKERS["kcf"]()
            tracker.init(resized_frame, initBB)
            is_not_detected_before = False
            break
    else:
        frame_resize = imutils.resize(frame, width=500)
        (success, box) = tracker.update(frame_resize)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            x1 = int(x * scale_factor)
            y1 = int(y * scale_factor)
            x2 = int((x + w) * scale_factor)
            y2 = int((y + h) * scale_factor)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 'Hand TRACKED', (x1, y2 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
        else:
            is_not_detected_before = True
    cv2.putText(frame, 'FPS: {:.2f}'.format(fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, 'Center: {}'.format(center_x), (540, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
    cv2.putText(frame, 'size: {}'.format(size), (540, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (233, 100, 25))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cap.release()
cv2.destroyAllWindows()