from __future__ import annotations

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from rs_tools._src.datamodule.components.base import BaseDataset

class MultiDataModule(LightningDataModule):
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
            train_shuffle  # turn off when performing regional inference
        )

        self.train_dataset = BaseDataset(
            datasets_spec=datasets_spec,
            split_file=split_file,
            split="train",
            transforms=transforms,
        )

        self.test_dataset = BaseDataset(
            datasets_spec=datasets_spec,
            split_file=split_file,
            split="test",
            transforms=transforms,
        )

        self.val_dataset = BaseDataset(
            datasets_spec=datasets_spec,
            split_file=split_file,
            split="val",
            transforms=transforms,
        )

    def prepare_data(self):
        self.train_dataset.prepare_data()
        self.test_dataset.prepare_data()
        self.val_dataset.prepare_data()

    def setup(self, stage):
        self.train_dataset.setup(stage)
        self.test_dataset.setup(stage)
        self.val_dataset.setup(stage)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.train_shuffle,
            persistent_workers=True,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=self.hparams.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=True,
            prefetch_factor=self.hparams.prefetch_factor,
        )