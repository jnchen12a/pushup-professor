from collections import deque

class AngleHolder:
    def __init__(self) -> None:
        self.shoulderToHip = deque(maxlen=5)
        self.hipToKnee = deque(maxlen=5)
        self.kneeToAnkle = deque(maxlen=5)

    def addShoulderToHip(self, angle: float) -> None:
        self.shoulderToHip.append(angle)

    def addHipToKnee(self, angle: float) -> None:
        self.hipToKnee.append(angle)

    def addKneeToAnkle(self, angle: float) -> None:
        self.kneeToAnkle.append(angle)

    def getShoulderToHip(self) -> float:
        return sum(self.shoulderToHip) / len(self.shoulderToHip) if len(self.shoulderToHip) != 0 else 0
    
    def getHipToKnee(self) -> float:
        return sum(self.hipToKnee) / len(self.hipToKnee) if len(self.hipToKnee) != 0 else 0
    
    def getKneeToAnkle(self) -> float:
        return sum(self.kneeToAnkle) / len(self.kneeToAnkle) if len(self.kneeToAnkle) != 0 else 0
    
    def addAll(self, shoulderToHipAngle: float, hipToKneeAngle: float, kneeToAnkleAngle: float) -> None:
        self.addShoulderToHip(shoulderToHipAngle)
        self.addHipToKnee(hipToKneeAngle)
        self.addKneeToAnkle(kneeToAnkleAngle)

    def getAll(self) -> tuple[float, float, float]:
        return self.getShoulderToHip(), self.getHipToKnee(), self.getKneeToAnkle()
    
    def __getitem__(self, key: int) -> float:
        if key == 0:
            return self.getShoulderToHip()
        elif key == 1:
            return self.getHipToKnee()
        elif key == 2:
            return self.getKneeToAnkle()
        else:
            raise IndexError(f'{key} is not a valid index.')