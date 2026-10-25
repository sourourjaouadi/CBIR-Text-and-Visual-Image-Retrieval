"""Compact HOG descriptor."""
import cv2
import numpy as np
from skimage.feature import hog
from backend.app.core.descriptors.base import Descriptor

class HOGDescriptor(Descriptor):
    name = "hog"
    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(cv2.resize(image, (64, 64)), cv2.COLOR_BGR2GRAY)
        return hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True).astype(np.float32)
