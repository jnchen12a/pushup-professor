import cv2 as cv

def checkStartRep(angles: list) -> bool:
    shoulderToElbow = angles[0] > -84 and angles[0] < -75
    elbowToWrist = True # angles[1] > -105 and angles[1] < -99
    shoulderToHip = True # angles[2] > -22 and angles[2] < -16
    hipToKnee = angles[3] > -27 and angles[3] < -22
    kneeToAnkle = angles[4] > -17 and angles[4] < -9

    return shoulderToElbow and elbowToWrist and shoulderToHip and hipToKnee and kneeToAnkle

def checkStartRep2(angles: list) -> bool:
    shoulderToHip = angles[0] > -25 and angles[0] < -13
    hipToKnee = angles[1] > -30 and angles[1] < -20
    kneeToAnkle = angles[2] > -20 and angles[2] < -5

    return shoulderToHip and hipToKnee and kneeToAnkle

def checkEndRep(angles: list) -> bool:
    shoulderToElbow = angles[0] > -28 and angles[0] < -9
    elbowToWrist = True # angles[1] > -138 and angles[1] < -130
    shoulderToHip = True # angles[2] > 1 and angles[2] < 3
    hipToKnee = angles[3] > -13 and angles[3] < -7
    kneeToAnkle = angles[4] > 5 and angles[4] < 11

    return shoulderToElbow and elbowToWrist and shoulderToHip and hipToKnee and kneeToAnkle

def checkEndRep2(angles: list) -> bool:
    shoulderToHip = angles[0] > -5 and angles[0] < 10
    hipToKnee = angles[1] > -15 and angles[1] < -5
    kneeToAnkle = angles[2] > 0 and angles[2] < 15

    return shoulderToHip and hipToKnee and kneeToAnkle

def writeRepsToScreen(img: cv.typing.MatLike, reps: int):
    text = str(reps)
    h, w = img.shape[:2]
    (text_width, text_height), baseline = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 5, 3)
    x = w - text_width - 500
    y = text_height + 5
    img = cv.putText(img, text, (0, y), cv.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 3)

    return img