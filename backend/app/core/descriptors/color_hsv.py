"""Normalized HSV color histogram descriptor."""
import cv2
import numpy as np
from backend.app.core.descriptors.base import Descriptor

class HSVHistogram(Descriptor):
    name = "color_hsv"
    metric = "l1"
    def extract(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).reshape(-1).astype(np.float32)
