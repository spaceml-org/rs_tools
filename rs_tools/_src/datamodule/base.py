from __future__ import annotations

import gc
import glob
import logging
import os
import random
from collections import Iterable
from typing import List, Union

import numpy as np
from dateutil.parser import parse
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm
import numpy as np
from typing import List
from loguru import logger
from torch.utils.data import Dataset
from rs_tools._src.preprocessing.normalize import apply_spectral_normalizer, apply_coordinate_normalizer
import xarray as xr
from torch.utils.data import Dataset, DataLoader, RandomSampler

# TODO: Copied from ITI repo and to be replaced
class BaseDataset(Dataset):

    def __init__(self, 
                 data: Union[str, list], 
                 editors: List[Editor], 
                 ext: str = None, 
                 limit: int = None,
                 months: list = None, 
                 date_parser=None, 
                 **kwargs):
        if isinstance(data, str):
            pattern = '*' if ext is None else '*' + ext
            data = sorted(glob.glob(os.path.join(data, "**", pattern), recursive=True))
        assert isinstance(data, Iterable), 'Dataset requires list of samples or path to files!'
        if months:  # assuming filename is parsable datetime
            if date_parser is None:
                date_parser = lambda f: parse(os.path.basename(f).split('.')[0])
            data = [d for d in data if date_parser(d).month in months]
        if limit is not None:
            data = random.sample(list(data), limit)
        self.data = data
        self.editors = editors

        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, _ = self.getIndex(idx)
        return data

    def sample(self, n_samples):
        it = DataLoader(self, batch_size=1, shuffle=True, num_workers=4).__iter__()
        samples = []
        while len(samples) < n_samples:
            try:
                samples.append(next(it).detach().numpy()[0])
            except Exception as ex:
                logging.error(str(ex))
                continue
        del it
        return np.array(samples)

    def getIndex(self, idx):
        try:
            return self.convertData(self.data[idx])
        except Exception as ex:
            logging.error('Unable to convert %s: %s' % (self.data[idx], ex))
            raise ex

    def getId(self, idx):
        return os.path.basename(self.data[idx]).split('.')[0]

    def convertData(self, data):
        kwargs = {}
        for editor in self.editors:
            data, kwargs = editor.convert(data, **kwargs)
        return data, kwargs

    def addEditor(self, editor):
        self.editors.append(editor)