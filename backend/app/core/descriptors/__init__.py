"""Image descriptor implementations."""
from backend.app.core.descriptors.color_hsv import HSVHistogram
from backend.app.core.descriptors.lbp import LBPDescriptor
from backend.app.core.descriptors.hog import HOGDescriptor
from backend.app.core.descriptors.vgg16 import VGG16Descriptor
DESCRIPTORS = {item.name: item for item in (HSVHistogram(), LBPDescriptor(), HOGDescriptor(), VGG16Descriptor())}
MPEG7_PROVIDED_DESCRIPTORS = {
    "ColorLayout", "ColorStructure", "ScalableColor", "EdgeHistogram", "HomogeneousTexture"
}
AVAILABLE_DESCRIPTOR_NAMES = set(DESCRIPTORS) | MPEG7_PROVIDED_DESCRIPTORS
