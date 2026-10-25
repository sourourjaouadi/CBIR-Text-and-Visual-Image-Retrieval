"""Local Binary Pattern histogram descriptor."""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from backend.app.core.descriptors.base import Descriptor

class LBPDescriptor(Descriptor):
    name = "lbp"
    metric = "l1"
    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp, bins=np.arange(11), range=(0, 10), density=True)
        return np.nan_to_num(hist).astype(np.float32)
