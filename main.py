import numpy as np
import cv2 as cv
from ultralytics import YOLO
from pose_utils import combineLeftRight, writeAnglesToScreen, calcAngles, calcAngles2
from rep_counter import checkStartRep2, checkEndRep2, writeRepsToScreen, writeDataToScreen
import math, time

# next: do smoothing

def testModel(path = ''):
    model = YOLO("yolov8n-pose.pt", verbose=False)
    if path == '':
        results = model(source=0, stream=True, verbose=False)
    else:
        results = model(source=path, stream=True, verbose=False)

    down = False
    up = False
    numReps = 0
    for result in results:
        xy = result.keypoints.xy
        points = combineLeftRight(xy)

        img = result.plot()
        # img = result.orig_img
        for p in points:
            cv.circle(img, (p[0], p[1]), 5, (0, 0, 255), -1)
        
        angles = calcAngles2(points)
        
        if not up and not down:
            # start of rep
            if checkStartRep2(angles):
                down = True
        elif down and not up:
            # check at the bottom of rep
            if checkEndRep2(angles):
                down = False
                up = True
        elif up and not down:
            if checkStartRep2(angles):
                up = False
                down = False
                numReps += 1
        

        img = writeRepsToScreen(img, numReps)
        # img = writeAnglesToScreen2(img, angles)

        cv.imshow('frame', img)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cv.destroyAllWindows()

def debuggingSave():
    model = YOLO("yolov8n-pose.pt", verbose=False)

    results = model(source=0, stream=True, verbose=False)

    out = None

    for result in results:
        xy = result.keypoints.xy
        points = combineLeftRight(xy)

        # Start from annotated frame
        img = result.plot()

        if out is None:
            h, w = img.shape[:2]
            fourcc = cv.VideoWriter_fourcc(*"mp4v")
            out = cv.VideoWriter("output.mp4", fourcc, 10, (w, h))
            print("Writer opened:", out.isOpened())

        # Draw custom points
        for p in points:
            cv.circle(img, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)

        img = writeAnglesToScreen(img, points)

        # Save frame to video
        out.write(img)

        # Show frame
        cv.imshow("frame", img)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    out.release()
    cv.destroyAllWindows()

def finalModel():
    # only uses webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        quit()
    model = YOLO("yolov8n-pose.pt", verbose=False)

    down = False
    up = False
    numReps = 0
    prev = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        # inference
        inferenceStart = time.time()
        results = model(frame, verbose=False)
        inferenceEnd = time.time()
        inferenceLatency = (inferenceEnd - inferenceStart) * 1000
        result = results[0]
        xy = result.keypoints.xy
        points = combineLeftRight(xy)

        img = result.plot()
        # img = result.orig_img
        for p in points:
            cv.circle(img, (p[0], p[1]), 5, (0, 0, 255), -1)
        
        angles = calcAngles2(points)
        
        if not up and not down:
            # start of rep
            if checkStartRep2(angles):
                down = True
        elif down and not up:
            # check at the bottom of rep
            if checkEndRep2(angles):
                down = False
                up = True
        elif up and not down:
            if checkStartRep2(angles):
                up = False
                down = False
                numReps += 1
        
        # fps calcs
        curr = time.time()
        fps = 1 / (curr - prev)
        prev = curr

        img = writeDataToScreen(img, numReps, fps, inferenceLatency)
        # img = writeAnglesToScreen2(img, angles)

        cv.imshow('frame', img)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    # testModel('./vids/normalTrim.mp4')
    # debuggingSave()
    finalModel()