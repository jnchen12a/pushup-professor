import math
import cv2 as cv

def combineLeftRight(l: list) -> list:
    '''
    Combines left- and right-side body parts into one point\n
    0: shoulder
    1: elbow
    2: wrist
    3: hip/booty
    4: knee
    5: ankle

    returns: [ [int, int] ]
    '''
    if len(l) == 0:
        # there are no points of interest detected
        return [ [0, 0] for i in range(6)]
    ret = []
    pairs = [[5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    for pair in pairs:
        x = (l[0][pair[0]][0] + l[0][pair[1]][0]) / 2
        y = (l[0][pair[0]][1] + l[0][pair[1]][1]) / 2

        ret.append((int(x), int(y)))

    return ret

def _calcAngle(origin: tuple, point: tuple) -> float:
    '''
    Calculates angle between origin and point, returns angle in degrees.
    Treats positive x-axis as 0 degrees.
    '''
    # need to negate y because of cv coordinate system, has (0, 0) in top left corner
    # returns 0 if both points are (0, 0) (possibly due to the fact that no points of interest were found)
    x, y = point[0] - origin[0], -(point[1] - origin[1])

    return math.degrees(math.atan2(y, x))

def calcAngles(points: list) -> list:
    '''
    Calculates the angle between a list of 2D points.
    Returns a list of angles, respectively
    Input should be [ [x, y], [x, y] ]
    '''
    ret = []
    for i in range(1, len(points)):
        ret.append(_calcAngle(points[i - 1], points[i]))

    return ret

def calcAngles2(points: list) -> list:
    pairsOfInterest = [ [0, 3], [3, 4], [4, 5] ]
    ret = []
    for pair in pairsOfInterest:
        ret.append(_calcAngle(points[pair[0]], points[pair[1]]))

    return ret

def writeAnglesToScreen(img: cv.typing.MatLike, angles: list) -> cv.typing.MatLike:
    '''
    Writes angles between selected body parts to screen, for debugging.
    '''
    pairs = [
        "sh to el",
        "el to wr",
        "sh to hip",
        "hip to kn",
        "kn to ank"
    ]
    yCoord = 0
    for i, name in enumerate(pairs):
        t = f'{name}: {angles[i]:.2f}'
        info = cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, 1.5, 2)
        yOffset = info[0][1]
        yCoord += yOffset + 5
        img = cv.putText(img, t, (0, yCoord), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    return img

def writeAnglesToScreen2(img: cv.typing.MatLike, angles: list) -> cv.typing.MatLike:
    pairs = [
        "sh to hip",
        "hip to kn",
        "kn to ank"
    ]
    yCoord = 0
    for i, name in enumerate(pairs):
        t = f'{name}: {angles[i]:.2f}'
        info = cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, 1.5, 2)
        yOffset = info[0][1]
        yCoord += yOffset + 5
        img = cv.putText(img, t, (0, yCoord), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    return img