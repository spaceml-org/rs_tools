from iti.data.editor import Editor
from abc import ABC, abstractmethod

from torchvision.transforms import (
    Compose,
    Lambda,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToPILImage,
    ToTensor,
)

# Editors that already exist (but not for dictionaries)
# - NormalizeEditor / ImageNormalizeEditor]
# - NanEditor

# TODO: Potential Transformations
# - Normalize data
# - Normalize coordinates
# - Reorder Bands
# - Band Selection
# - Nan mask
# - Unit conversion

# NOTE: Maybe not needed?
class BandOrderEditor(Editor):
    def call(self, data, **kwargs):
        pass

class BandSelectionEditor(Editor):
    def call(self, data, **kwargs):
        pass

class NanMaskEditor(Editor):
    def call(self, data, **kwargs):
        pass      

class CoordNormEditor(Editor):
    def call(self, data, **kwargs):
        pass  
class RadUnitEditor(Editor):
    def call(self, data, **kwargs):
        pass




