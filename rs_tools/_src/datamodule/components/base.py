from __future__ import annotations

import numpy as np
from loguru import logger
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(
        self,
        datasets_spec,
        datasets_dir,
        split,
        tile_format,
        transforms=None
    ):
        # TODO: Check that dataset directory exists
        # TODO: List all files in the dataset directory

        if not tile_format in ["nc", "tif", "npy"]:
            raise ValueError(
                f"invalid tile format '{tile_format}', must be 'nc', 'tif' or 'npy'."
            )

        if not split in ["test", "train", "val"]:
            raise ValueError(
                f"invalid split '{split}', must be one of 'train', 'test' or 'val'"
            )
        # if not present skip_constant_channels is set to false
        for v in datasets_spec.values():
            if not "skip_constant_channels" in v.keys():
                v["skip_constant_channels"] = False

        check = (
            (isinstance(datasets_spec, dict) or isinstance(datasets_spec, DictConfig))
            and (
                np.alltrue([isinstance(i, dict) for i in datasets_spec.values()])
                or np.alltrue(
                    [isinstance(i, DictConfig) for i in datasets_spec.values()]
                )
            )
            and np.alltrue(
                [
                    set(i.keys()) == {"class", "skip_constant_channels", "kwargs"}
                    for i in datasets_spec.values()
                ]
            )
        )

        if not check:
            raise ValueError(
                "'datasets_spec' must be a list of dicts with keys ['class', 'kwargs', 'skip_constant_channels']"
            )

        self.datasets_spec = datasets_spec
        self.transforms = transforms

        # if classes are strings with class names, then eval them
        for d in self.datasets_spec.values():
            if isinstance(d["class"], str):
                d["class"] = utils.get_class(d["class"])

        # instantiate datasets
        self.datasets = {
            k: d["class"](split_file=split_file, split=split, **d["kwargs"])
            for k, d in self.datasets_spec.items()
        }

    def prepare_data(self):
        pass

    def setup(self, stage):
        pass

    def __len__(self):
        pass

    def __repr__(self):
        return "/".join([str(d) for d in self.datasets.values()])

    def __getitem__(self, idx):
        pass