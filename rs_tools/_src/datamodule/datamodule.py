from __future__ import annotations

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from rs_tools._src.datamodule.components.base import BaseDataset

class ITIDataModule(LightningDataModule):
    def __init__(
        self,
        datasets_spec,
        split_file,
        transforms=None,
        batch_size: int = 16,
        num_workers: int = 1,
        pin_memory: bool = False,
        prefetch_factor: int = 4,
        train_shuffle=True,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.train_shuffle = (
            train_shuffle
        )

        # TODO: Add function to filter train/test/val split
        # Filter data for months
        # train_list = [training files]

        # TODO: Calculate mean/std for training set
        # Go through all files and calculate mean/stdev

        self.train_A = BaseDataset(...)
        self.train_B = BaseDataset(...)
        self.val_A = BaseDataset(...)
        self.val_B = BaseDataset(...)

    # def prepare_data(self):
    #     self.train_dataset.prepare_data()
    #     self.test_dataset.prepare_data()
    #     self.val_dataset.prepare_data()

    def train_dataloader(self):
        gen_A = DataLoader(dataset=self.train_A, ...)
        gen_B = DataLoader(dataset=self.train_B, ...)
        dis_A = DataLoader(dataset=self.train_A, ...)
        dis_B = DataLoader(dataset=self.train_B, ...)
        # return DataLoader(
        #     dataset=self.train_dataset,
        #     batch_size=self.hparams.batch_size,
        #     num_workers=self.hparams.num_workers,
        #     pin_memory=self.pin_memory,
        #     shuffle=self.train_shuffle,
        #     persistent_workers=True,
        #     prefetch_factor=self.hparams.prefetch_factor,
        # )
        return {"gen_A": gen_A, "dis_A": dis_A, "gen_B": gen_B, "dis_B": dis_B}
    
    def val_dataloader(self):
            A = DataLoader(self.A_valid, ...)
            B = DataLoader(self.B_valid, ...)
            return [A, B]