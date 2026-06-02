import cv2 as cv

def checkStartRep(angles: list) -> bool:
    shoulderToElbow = angles[0] > -84 and angles[0] < -75
    elbowToWrist = angles[1] > -105 and angles[1] < -99
    shoulderToHip = angles[2] > -22 and angles[2] < -16
    hipToKnee = angles[3] > -27 and angles[3] < -22
    kneeToAnkle = angles[4] > -17 and angles[4] < -9

    return shoulderToElbow and elbowToWrist and shoulderToHip and hipToKnee and kneeToAnkle

def checkEndRep(angles: list) -> bool:
    shoulderToElbow = angles[0] > -28 and angles[0] < -9
    elbowToWrist = angles[1] > -138 and angles[1] < -130
    shoulderToHip = angles[2] > 1 and angles[2] < 3
    hipToKnee = angles[3] > -13 and angles[3] < -7
    kneeToAnkle = angles[4] > 5 and angles[4] < 11

    return shoulderToElbow and elbowToWrist and shoulderToHip and hipToKnee and kneeToAnkle

def writeRepsToScreen(img: cv.typing.MatLike, reps: int):
    text = str(reps)
    h, w = img.shape[:2]
    (text_width, text_height), baseline = cv.getTextSize(
    text, cv.FONT_HERSHEY_SIMPLEX, 3, 2)
    x = w - text_width - 5
    y = text_height + 5
    print(f"Drawing '{text}' at ({x}, {y})")
    print(w, h)
    cv.putText(img, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 2)

    return img