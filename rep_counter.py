import cv2 as cv
from angleHolder import AngleHolder

def checkStartRep(angles: list) -> bool:
    shoulderToElbow = angles[0] > -84 and angles[0] < -75
    elbowToWrist = True # angles[1] > -105 and angles[1] < -99
    shoulderToHip = True # angles[2] > -22 and angles[2] < -16
    hipToKnee = angles[3] > -27 and angles[3] < -22
    kneeToAnkle = angles[4] > -17 and angles[4] < -9

    return shoulderToElbow and elbowToWrist and shoulderToHip and hipToKnee and kneeToAnkle

def checkStartRep2(angleHolder: AngleHolder) -> bool:
    shoulderToHip = angleHolder.getShoulderToHip() > -25 and angleHolder.getShoulderToHip() < -13
    hipToKnee = angleHolder.getHipToKnee() > -30 and angleHolder.getHipToKnee() < -20
    kneeToAnkle = angleHolder.getKneeToAnkle() > -20 and angleHolder.getKneeToAnkle() < -5

    return shoulderToHip and hipToKnee and kneeToAnkle

def checkEndRep(angles: list) -> bool:
    shoulderToElbow = angles[0] > -28 and angles[0] < -9
    elbowToWrist = True # angles[1] > -138 and angles[1] < -130
    shoulderToHip = True # angles[2] > 1 and angles[2] < 3
    hipToKnee = angles[3] > -13 and angles[3] < -7
    kneeToAnkle = angles[4] > 5 and angles[4] < 11

    return shoulderToElbow and elbowToWrist and shoulderToHip and hipToKnee and kneeToAnkle

def checkEndRep2(angleHolder: AngleHolder) -> bool:
    shoulderToHip = angleHolder.getShoulderToHip() > -5 and angleHolder.getShoulderToHip() < 10
    hipToKnee = angleHolder.getHipToKnee() > -15 and angleHolder.getHipToKnee() < -5
    kneeToAnkle = angleHolder.getKneeToAnkle() > 0 and angleHolder.getKneeToAnkle() < 15

    return shoulderToHip and hipToKnee and kneeToAnkle

def writeRepsToScreen(img: cv.typing.MatLike, reps: int) -> cv.typing.MatLike:
    text = str(reps)
    h, w = img.shape[:2]
    (text_width, text_height), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 5, 3)
    x = w - text_width - 500
    y = text_height + 5
    img = cv.putText(img, text, (0, y), cv.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3)

    return img

def writeDataToScreen(img: cv.typing.MatLike, reps: int, fps: float, latency: float) -> cv.typing.MatLike:
    y = 0
    dataDict = {
        'Reps': reps,
        'FPS': round(fps),
        "Latency": latency
    }

    for label, num in dataDict.items():
        t = f'{label}: {num:.2f}' if isinstance(num, float) else f'{label}: {num}'
        t += 'ms' if label == 'Latency' else ''
        info = cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, 0.75, 1)
        yOffset = info[0][1]
        y += yOffset + 5
        img = cv.putText(img, t, (0, y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)

    return img