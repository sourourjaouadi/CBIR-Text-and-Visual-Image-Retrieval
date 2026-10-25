"""Small offline contract test for VGG16 vector shape and normalization."""
import numpy as np
import pytest


def test_vgg16_feature_shape_unit_norm_and_identical_distance(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from backend.app.core.descriptors.vgg16 import VGG16Descriptor

    descriptor = VGG16Descriptor(device="cpu", layer="fc7", model=object())

    def deterministic_inference(batch):
        # A compact stand-in avoids allocating VGG16's large classifier in unit tests.
        pooled = batch.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return pooled.expand(-1, 4096)

    monkeypatch.setattr(descriptor, "_infer", deterministic_inference)
    image = np.full((64, 48, 3), (20, 90, 180), dtype=np.uint8)
    first = descriptor.extract(image)
    second = descriptor.extract(image.copy())

    assert first.shape == (4096,)
    assert first.dtype == np.float32
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-5)
    assert descriptor.distance(first, second) == pytest.approx(0.0, abs=1e-6)
