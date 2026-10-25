"""ImageNet VGG16 features for deep visual retrieval.

The expensive torchvision model is initialized lazily and shared by all descriptor
instances. ``extract_batch`` returns only valid images and their source paths, which
lets the indexing job skip unreadable files while preserving row/ID alignment.
"""
from __future__ import annotations

from functools import lru_cache
import logging
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from backend.app.core.descriptors.base import Descriptor

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _get_model(device_name: str):
    """Load pretrained VGG16 once for the requested feature layer and device."""
    import torch
    from torchvision.models import VGG16_Weights, vgg16

    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(torch.device(device_name))
    return model


class _ImagePathDataset:
    """Top-level picklable dataset so DataLoader workers work on Windows too."""
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image as PILImage
        path = self.paths[index]
        try:
            with PILImage.open(path) as image:
                return path, self.transform(image.convert("RGB"))
        except Exception as exc:
            logger.warning("Skipping unreadable VGG16 input %s: %s", path, exc)
            return path, None


def _collate_images(items):
    import torch
    valid = [(path, tensor) for path, tensor in items if tensor is not None]
    if not valid:
        return [], None
    return [path for path, _ in valid], torch.stack([tensor for _, tensor in valid])


class VGG16Descriptor(Descriptor):
    """L2-normalized VGG16 fc7 (4096-D) or pooled conv5 (512-D) features."""

    name = "vgg16"
    metric = "l2"

    def __init__(self, layer: str | None = None, device: str | None = None,
                 num_workers: int | None = None, model=None):
        self.layer = layer or os.getenv("VGG_LAYER", "fc7")
        if self.layer not in {"fc7", "conv5_pool"}:
            raise ValueError("VGG_LAYER must be 'fc7' or 'conv5_pool'")
        self.device_name = device or ("cuda" if self._cuda_available() else "cpu")
        self.num_workers = int(num_workers if num_workers is not None else os.getenv("VGG_NUM_WORKERS", "0"))
        self._model_override = model
        self._transform = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    @property
    def model(self):
        if self._model_override is not None:
            return self._model_override
        return _get_model(self.device_name)

    def _preprocessor(self):
        if self._transform is None:
            from torchvision import transforms
            self._transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        return self._transform

    def _infer(self, batch):
        import torch
        import torch.nn.functional as functional
        model = self.model
        with torch.no_grad():
            if self.layer == "conv5_pool":
                result = functional.adaptive_avg_pool2d(model.features(batch), 1).flatten(1)
            else:
                features = model.features(batch)
                flattened = model.avgpool(features).flatten(1)
                result = model.classifier[:5](flattened)
        return result

    def _to_tensor(self, image: np.ndarray):
        from PIL import Image as PILImage
        import cv2
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("VGG16 expects a grayscale, BGR, or BGRA image array")
        return self._preprocessor()(PILImage.fromarray(rgb))

    @staticmethod
    def _normalize(features) -> np.ndarray:
        import torch.nn.functional as functional
        normalized = functional.normalize(features.float(), p=2, dim=1)
        return normalized.cpu().numpy().astype(np.float32, copy=False)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract one OpenCV BGR image as a unit-normalized float32 vector."""
        import torch
        tensor = self._to_tensor(image).unsqueeze(0).to(self.device_name)
        return self._normalize(self._infer(tensor))[0]

    def extract_batch(self, list_of_paths: Iterable[str | Path], batch_size: int = 64
                      ) -> tuple[np.ndarray, list[Path]]:
        """Extract valid paths in batches; return vectors and corresponding paths.

        TODO: add persistent worker tuning and distributed extraction for multi-GPU jobs.
        """
        from torch.utils.data import DataLoader

        paths = [Path(path) for path in list_of_paths]
        transform = self._preprocessor()
        descriptor = self

        loader = DataLoader(_ImagePathDataset(paths, transform), batch_size=max(1, int(batch_size)), shuffle=False,
                            num_workers=max(0, self.num_workers), collate_fn=_collate_images)
        chunks: list[np.ndarray] = []
        valid_paths: list[Path] = []
        for batch_paths, tensors in loader:
            if tensors is None:
                continue
            features = descriptor._infer(tensors.to(self.device_name))
            chunks.append(descriptor._normalize(features))
            valid_paths.extend(batch_paths)
        dimension = 4096 if self.layer == "fc7" else 512
        return (np.concatenate(chunks, axis=0) if chunks else np.empty((0, dimension), np.float32),
                valid_paths)
