from iti.data.editor import Editor

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

# TODO: Add transform to organize bands by wavelength

# TODO: Potential Transformations
# - Normalize (of course)
# - Reorder Bands
# - Random Flipping (Maybe?)
# - Random Brightness Contrast
# - Nan mask
# - Unit conversion

class OrderBandEditor(Editor):
    def call(self, data, **kwargs):
        pass


