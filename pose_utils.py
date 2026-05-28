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

def calcAngle(origin: tuple, point: tuple) -> float:
    '''
    Calculates angle between origin and point, returns angle in degrees.
    Treats positive x-axis as 0 degrees.
    '''
    # need to negate y because of cv coordinate system, has (0, 0) in top left corner
    # returns 0 if both points are (0, 0) (possibly due to the fact that no points of interest were found)
    x, y = point[0] - origin[0], -(point[1] - origin[1])

    return math.degrees(math.atan2(y, x))

def writeAnglesToScreen(img: cv.typing.MatLike, points: list) -> cv.typing.MatLike:
    '''
    Writes angles between selected body parts to screen, for debugging.
    '''
    pairs = {
        "sh to el": (0, 1),
        "el to wr": (1, 2),
        "sh to hip": (0, 3),
        "hip to kn": (3, 4),
        "kn to ank": (4, 5)
    }
    yCoord = 0
    for name, p in pairs.items():
        t = f'{name}: {calcAngle(points[p[0]], points[p[1]]):.2f}'
        info = cv.getTextSize(t, cv.FONT_HERSHEY_SIMPLEX, 1.5, 2)
        yOffset = info[0][1]
        yCoord += yOffset + 5
        cv.putText(img, t, (0, yCoord), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    return img