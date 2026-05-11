def combineLeftRight(l: list) -> list:
    '''
    0: shoulder
    1: elbow
    2: wrist
    3: hip/booty
    4: knee
    5: ankle

    returns: [ [int, int] ]
    '''
    ret = []
    # [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    pairs = [[5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
    for pair in pairs:
        x = (l[0][pair[0]][0] + l[0][pair[1]][0]) / 2
        y = (l[0][pair[0]][1] + l[0][pair[1]][1]) / 2

        ret.append((int(x), int(y)))

    return ret