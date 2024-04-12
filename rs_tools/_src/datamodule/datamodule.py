from __future__ import annotations

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from rs_tools._src.datamodule.components.base import BaseDataset
from rs_tools._src.utils.io import get_files

class ITIDataModule(LightningDataModule):
    def __init__(
        self,
        datasets_spec,
        batch_size: int = 4,
        iterations_per_epoch=1e4,
        num_workers: int = 1,
        train_shuffle=True,
    ):
        super().__init__()
        self.datasets_spec = datasets_spec
        self.batch_size = batch_size
        self.iterations_per_epoch = iterations_per_epoch
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle

        if not isinstance(datasets_spec, dict):
            raise ValueError("datasets_spec must be a dictionary")
        
        datasets = datasets_spec.keys()
        if len(datasets) != 2:
            raise ValueError("datasets_spec must contain two datasets")
        
        # Get filenames from dataset specs
        self.A_filenames = get_files(datasets_spec=datasets_spec[datasets[0]])
        self.B_filenames = get_files(datasets_spec=datasets_spec[datasets[1]])

        # TODO: Add function to filter train/test/val split
        # Filter data for months
        # train_list = [training files]

        # TODO: Calculate mean/std for training set
        # Go through all files and calculate mean/stdev

        self.A_train_ds = BaseDataset(...)
        self.B_train_ds = BaseDataset(...)
        self.A_valid_ds = BaseDataset(...)
        self.B_valid_ds = BaseDataset(...)

    def train_dataloader(self):
        gen_A = DataLoader(dataset=self.A_train_ds,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=self.train_shuffle,
                           sampler=RandomSampler(self.A_train_ds, replacement=True, num_samples=self.iterations_per_epoch)
        )
        dis_A = DataLoader(dataset=self.A_train_ds,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=self.train_shuffle,
                            sampler=RandomSampler(self.A_train_ds, replacement=True, num_samples=self.iterations_per_epoch)
          )
        gen_B = DataLoader(dataset=self.B_train_ds,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers,
                           shuffle=self.train_shuffle,
                           sampler=RandomSampler(self.B_train_ds, replacement=True, num_samples=self.iterations_per_epoch)
        )
        dis_B = DataLoader(dataset=self.B_train_ds,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            shuffle=self.train_shuffle,
                            sampler=RandomSampler(self.B_train_ds, replacement=True, num_samples=self.iterations_per_epoch)
          )
        return {"gen_A": gen_A, "dis_A": dis_A, "gen_B": gen_B, "dis_B": dis_B}
    
    def val_dataloader(self):
            A = DataLoader(self.A_valid_ds, 
                           batch_size=self.batch_size, 
                           num_workers=self.num_workers,
                           shuffle=False)
            B = DataLoader(self.B_valid_ds, 
                           batch_size=self.batch_size, 
                           num_workers=self.num_workers,
                           shuffle=False)
            return [A, B]