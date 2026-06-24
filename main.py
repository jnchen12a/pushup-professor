import numpy as np
import cv2 as cv
from ultralytics import YOLO
from pose_utils import combineLeftRight, writeAnglesToScreen, calcAngles, calcAngles2, writeAnglesToScreen2
from rep_counter import checkStartRep2, checkEndRep2, writeRepsToScreen, writeDataToScreen
import math, time, os
from angleHolder import AngleHolder
from collections import deque
from io import TextIOWrapper
from argparse import ArgumentParser

# next: documentation, graphs, demo (with skeleton), demo (get data)

latencies = deque(maxlen=100)
PROGRAM_START = time.perf_counter()

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

def initLoggingFile() -> None:
    with open('./log.txt', 'w') as file:
        file.write('Frame,Timestamp,Capture (ms),Inference (ms),Angles (ms),Smooth (ms),Count (ms),Current Latency (ms),Average Latency (ms),Std Dev Latency (ms),FPS\n')
        file.flush()

def printFPSStats(frame: int, capture: float, inference: float, angles: float, smooth: float, count: float) -> None:
    print(f'Capture: {capture:.2f} ms')
    print(f'Inference: {inference:.2f} ms')
    print(f'Angles: {angles:.2f} ms')
    print(f'Smooth: {smooth:.2f} ms')
    print(f'Count: {count:.2f} ms')
    s = capture + inference + angles + smooth + count
    print()
    print(f'FPS: {1 / (s / 1000)}')
    print()
    print('Latency (last 100 frames)')
    print(f'Current: {s:.2f} ms')
    latencies.append(s)
    avg = np.mean(latencies)
    stdDev = np.std(latencies)
    print(f'Average: {avg:.2f} ms')
    print(f'Std Dev: {stdDev:.2f} ms')

def logFPSStats(frame: int, capture: float, inference: float, angles: float, smooth: float, count: float, file: TextIOWrapper) -> None:
    s = capture + inference + angles + smooth + count
    avg = np.mean(latencies)
    stdDev = np.std(latencies)
    file.write(f'{frame},{time.perf_counter() - PROGRAM_START},{capture:.2f},{inference:.2f},{angles:.2f},{smooth:.2f},{count:.2f},{s:.2f},{avg:.2f},{stdDev:.2f},{1 / (s / 1000)}\n')

def finalModel(imgsz: int, skipCount: int, log: bool) -> None:
    # only uses webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        quit()
    model = YOLO("yolov8n-pose.pt", verbose=False)

    if log:
        initLoggingFile()
        file = open('./log.txt', 'a')

    down = False
    up = False
    numReps = 0
    aglHolder = AngleHolder()
    f = 0
    skipCount += 1 # so that we actually skip skipCount frames before running inference again

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()

        if not ret:
            print("Error: Failed to grab frame.")
            break

        # inference
        runInference = (f % skipCount == 0)
        if runInference:
            results = model(frame, imgsz=imgsz, verbose=False)
        
        t2 = time.perf_counter()
        result = results[0]
        xy = result.keypoints.xy
        points = combineLeftRight(xy)

        img = result.plot()
        # img = result.orig_img
        for p in points:
            cv.circle(img, (p[0], p[1]), 5, (0, 0, 255), -1)
        
        t3 = time.perf_counter()
        angles = calcAngles2(points)
        t4 = time.perf_counter()
        aglHolder.addAll(*angles)
        t5 = time.perf_counter()
        
        if not up and not down:
            # start of rep
            if checkStartRep2(aglHolder):
                down = True
        elif down and not up:
            # check at the bottom of rep
            if checkEndRep2(aglHolder):
                down = False
                up = True
        elif up and not down:
            if checkStartRep2(aglHolder):
                up = False
                down = False
                numReps += 1
        t6 = time.perf_counter()
        
        # fps calcs
        captureMs = (t1 - t0) * 1000
        inferenceMs = (t2 - t1) * 1000
        anglesMs = (t4 - t3) * 1000
        smoothMs = (t5 - t4) * 1000
        repCountMs = (t6 - t5) * 1000

        os.system('cls')
        printFPSStats(f, captureMs, inferenceMs, anglesMs, smoothMs, repCountMs)
        if log:
            logFPSStats(f, captureMs, inferenceMs, anglesMs, smoothMs, repCountMs, file)

        # img = writeDataToScreen(img, numReps, fps, inferenceLatency)
        img = writeAnglesToScreen2(img, aglHolder)

        cv.imshow('frame', img)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

        f += 1

    cv.destroyAllWindows()
    file.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--imgsz', help='Dimensions of image which model takes in.', type=int, default=640)
    parser.add_argument('--skip-frames', help='Number of frames we skip each time before running inference.', type=int, default=0)
    parser.add_argument('--log', help='Toggle logging.', action='store_true')

    args = parser.parse_args()
    # testModel('./vids/normalTrim.mp4')
    # debuggingSave()
    finalModel(imgsz=args.imgsz, skipCount=args.skip_frames, log=args.log)