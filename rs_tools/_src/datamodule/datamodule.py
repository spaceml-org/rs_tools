from __future__ import annotations

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
from rs_tools._src.datamodule.dataset import BaseDataset
from rs_tools._src.utils.io import get_files
from rs_tools._src.datamodule.utils import split_dataset
from rs_tools._src.preprocessing.normalize import normalize

class ITIDataModule(LightningDataModule):
    def __init__(
        self,
        datasets_spec: dict,
        include_coords: bool,
        include_cloudmask: bool,
        include_nanmask: bool,
        normalize_coords: bool,
        datasets_split: dict,
        batch_size: int=4,
        iterations_per_epoch: int=1e4,
        num_workers: int=1,
        train_shuffle: bool=True,
    ):
        super().__init__()
        self.datasets_spec = datasets_spec
        self.datasets_split = datasets_split

        self.include_coords = include_coords
        self.include_cloudmask = include_cloudmask
        self.include_nanmask = include_nanmask
        self.normalize_coords = normalize_coords

        self.batch_size = batch_size
        self.iterations_per_epoch = iterations_per_epoch
        self.num_workers = num_workers
        self.train_shuffle = train_shuffle

        # extract dataset specifictions from dataloader config
        self.datasets = self.datasets_spec.keys()
        if len(self.datasets) != 2:
            raise ValueError("datasets_spec must contain two datasets")
        
    def setup(self):
        # Get filenames from dataset specs
        self.A_filenames = get_files(datasets_spec=self.datasets_spec[self.datasets[0]])
        self.B_filenames = get_files(datasets_spec=self.datasets_spec[self.datasets[1]])

        # split files based on train/test/val criteria
        self.A_filenames_train, self.A_filenames_valid = split_dataset(
             filenames=self.A_filenames, 
             split_spec=self.datasets_split[self.datasets[0]])
        self.B_filenames_train, self.B_filenames_valid = split_dataset(
             filenames=self.B_filenames, 
             split_spec=self.datasets_split[self.datasets[1]])
        
        # Prepare data & normalize
        ds_norm_A, ds_norm_B = self.prepare_data()
        
        # Create datasets
        self.A_train_ds = BaseDataset(
            file_list=self.A_filenames_train,
            bands=self.datasets_spec[self.datasets[0]]["bands"],
            transforms=self.datasets_spec[self.datasets[0]]["transforms"]
            include_coords=self.include_coords,
            include_cloudmask=self.include_cloudmask,
            include_nanmask=self.include_nanmask,
            band_norm=ds_norm_A,
            coord_norm=self.normalize_coords,
        )
        self.B_train_ds = BaseDataset(
            file_list=self.B_filenames_train,
            bands=self.datasets_spec[self.datasets[1]]["bands"],
            transforms=self.datasets_spec[self.datasets[1]]["transforms"]
            include_coords=self.include_coords,
            include_cloudmask=self.include_cloudmask,
            include_nanmask=self.include_nanmask,
            band_norm=ds_norm_B,
            coord_norm=self.normalize_coords,
        )   
        self.A_valid_ds = BaseDataset(
            file_list=self.A_filenames_valid,
            bands=self.datasets_spec[self.datasets[0]]["bands"],
            transforms=self.datasets_spec[self.datasets[0]]["transforms"]
            include_coords=self.include_coords,
            include_cloudmask=self.include_cloudmask,
            include_nanmask=self.include_nanmask,
            band_norm=ds_norm_A,
            coord_norm=self.normalize_coords,
        )   
        self.B_valid_ds = BaseDataset(
            file_list=self.B_filenames_valid,
            bands=self.datasets_spec[self.datasets[1]]["bands"],
            transforms=self.datasets_spec[self.datasets[1]]["transforms"]
            include_coords=self.include_coords,
            include_cloudmask=self.include_cloudmask,
            include_nanmask=self.include_nanmask,
            band_norm=ds_norm_B,
            coord_norm=self.normalize_coords,
        )

    def prepare_data(self):
        """
        Calculate normalization based on training files
        """
        ds_norm_A = normalize(self.A_filenames_train)
        ds_norm_B = normalize(self.B_filenames_train)
        return ds_norm_A, ds_norm_B

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