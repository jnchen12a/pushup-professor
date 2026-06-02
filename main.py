import numpy as np
import cv2 as cv
from ultralytics import YOLO
from pose_utils import combineLeftRight, writeAnglesToScreen, calcAngles
from rep_counter import checkStartRep, checkEndRep, writeRepsToScreen
import math, time

# next: redo count reps logic, get reps on screen

def testModel():
    model = YOLO("yolov8n-pose.pt", verbose=False)
    results = model(source="./normal.MOV", stream=True, verbose=False)

    inMiddleOfRep = False
    numReps = 0
    for result in results:
        xy = result.keypoints.xy
        points = combineLeftRight(xy)

        annotated = result.plot()
        img = result.orig_img
        for p in points:
            cv.circle(img, (p[0], p[1]), 5, (0, 0, 255), -1)
        
        angles = calcAngles(points)
        
        if not inMiddleOfRep:
            if checkStartRep(angles):
                inMiddleOfRep = True
                print('rep started')
        else:
            if checkEndRep(angles):
                inMiddleOfRep = False
                numReps += 1
                print('rep ended')

        img = writeAnglesToScreen(img, angles)
        img = writeRepsToScreen(img, numReps)

        cv.imshow('frame', img)
        if cv.waitKey(1) == ord("q"):
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

        if cv.waitKey(1) == ord("q"):
            break

    out.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    testModel()
    # debuggingSave()