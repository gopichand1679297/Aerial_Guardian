import cv2
import numpy as np

class MotionCompensator:
    def __init__(self):
        self.prev_gray = None

    def compensate(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return frame

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        dx = np.mean(flow[..., 0])
        dy = np.mean(flow[..., 1])

        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        stabilized = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        self.prev_gray = gray

        return stabilized