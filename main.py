import numpy as np
import cv2 as cv
from ultralytics import YOLO
from pose_utils import combineLeftRight, calcAngle, writeAnglesToScreen
import math

# next: get degrees on screen count reps

def useWebcam():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: cannot open camera')
        return
    while True:
        ret, frame = cap.read()

        if not ret:
            print('ERROR: Cannot read frame???')
            break
        
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def testModel():
    model = YOLO("yolov8n-pose.pt")
    results = model(source=0, stream=True)

    for result in results:
        frame = result.plot()
        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord("q"):
            break

    cv.destroyAllWindows()

def testModelOneFrame():
    model = YOLO("yolov8n-pose.pt")
    results = model(source="./test.jpg", stream=False)

    for result in results:
        xy = result.keypoints.xy
        points = combineLeftRight(xy)

        annotated = result.plot()
        img = result.orig_img
        for p in points:
            cv.circle(img, (p[0], p[1]), 5, (0, 0, 255), -1)
        
        img = writeAnglesToScreen(img, points)

        cv.imshow('frame', img)
        cv.waitKey(0)

    cv.destroyAllWindows()


if __name__ == '__main__':
    testModelOneFrame()