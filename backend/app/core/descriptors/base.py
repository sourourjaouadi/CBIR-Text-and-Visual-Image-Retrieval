"""Descriptor interface shared by computed and provided features."""
from abc import ABC, abstractmethod
import numpy as np

class Descriptor(ABC):
    name: str
    metric: str = "l2"
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Return a finite, one-dimensional float32 feature vector."""
    def distance(self, left: np.ndarray, right: np.ndarray) -> float:
        delta = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
        if self.metric == "l1":
            return float(np.abs(delta).sum())
        return float(np.linalg.norm(delta))
