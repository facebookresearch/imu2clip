# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import random
from typing import Optional
import torch
import pytorch_lightning as pl
from dataset.ego4d.utils.utils import load_csv, load_json

from dataset.ego4d.dataloader import Ego4dDatasetSupervised, Ego4dDataset
from dataset.ego4d.dataloader_unsupervised import Ego4dDatasetUnsupervised


from dataset.ego4d.dataloader import collate_wrapper

random.seed(0)


class Split(object):
    def __init__(
        self,
        random_split: int = 0,
        split: str = "training",
        video_uid_sample_rate: float = 1.0,
    ):
        assert split in ["training", "validation", "test"]
        self.set = load_json(f"./splits/{split}_{random_split}.json")

        if video_uid_sample_rate != 1.0:
            self.scale(video_uid_sample_rate)

    def scale(self, video_uid_sample_rate: float):
        # Typically for debugging purposes, etc.
        assert video_uid_sample_rate < 1.0 and video_uid_sample_rate > 0.0
        n_videos_to_return = max(int(len(self.set) * video_uid_sample_rate), 1)
        print(f"Reducing to {n_videos_to_return} videos ...")
        self.set = set(list(self.set)[:n_videos_to_return])

    def filter(self, video_uid):
        # this video ids is problematic
        if video_uid in ["ec344610-74f4-4765-9c3f-0837ef78055d"]:
            return True
        return False


class Ego4dDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, pin_memory, drop_last, dataset_params):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.dataset_params = dataset_params

        self.prepare_data_per_node = False
        self._log_hyperparams = True

        self.collate_fn = lambda data: collate_wrapper(
            data, self.dataset_params["list_modalities"]
        )

        # Get splits ready
        self.filter_video_uids_train = Split(random_split=0, split="training")
        self.filter_video_uids_validation = Split(random_split=0, split="validation")
        self.filter_video_uids_test = Split(random_split=0, split="test")

    def get_dataset(
        self,
        split: str,
        video_uid_sample_rate: float = 1.0,
        window_sample_rate: float = 1.0,
        max_n_windows_per_video: Optional[int] = None,
    ) -> Ego4dDataset:

        if split == "training":
            filter_video_uids_split = self.filter_video_uids_train

        elif split == "validation":
            filter_video_uids_split = self.filter_video_uids_validation

        elif split == "test":
            filter_video_uids_split = self.filter_video_uids_test

        if video_uid_sample_rate != 1.0:
            filter_video_uids_split.scale(video_uid_sample_rate)

        return Ego4dDataset(
            window_sec=self.dataset_params["window_sec"],
            video="video" in self.dataset_params["list_modalities"],
            imu="imu" in self.dataset_params["list_modalities"],
            narr="text" in self.dataset_params["list_modalities"],
            audio="audio" in self.dataset_params["list_modalities"],
            return_tuple=False,
            target_frames_in_window=self.dataset_params["target_fps"],
            filter_video_uids=filter_video_uids_split.filter,
            clean_narration_func=self.dataset_params["clean_narration_func"]
            if self.dataset_params["clean_narration_func"]
            else lambda x: x,
            filter_narration_func=self.dataset_params["filter_narration_func"]
            if self.dataset_params["filter_narration_func"]
            else lambda x: True,
            window_sample_rate=window_sample_rate,
            max_n_windows_per_video=max_n_windows_per_video,
        )

    def setup(self, stage: Optional[str] = None):

        # Initialize data
        if stage in (None, "fit"):
            print("TRAIN")
            self.train = self.get_dataset("training")

            print("VALIDATION")
            self.val = self.get_dataset("validation")

        if stage in (None, "test"):
            print("TEST")
            self.test = self.get_dataset("test")
            self.predict = self.test

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )


class SupervisedEgo4dDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, pin_memory, drop_last, dataset_params):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.dataset_params = dataset_params

        self.prepare_data_per_node = False
        self._log_hyperparams = True

        # Get splits ready
        self.lable_dict = {
            "head movement": 0,
            "stands up": 1,
            "sits down": 2,
            "walking": 3,
        }
        self.n_classes = len(self.lable_dict)

    def get_dataset(
        self,
        split: str,
    ) -> Ego4dDatasetSupervised:
        path = f"./splits/dataset_motion_narr_2.5_{split}_0.csv"

        return Ego4dDatasetSupervised(
            window_sec=self.dataset_params["window_sec"],
            video="video" in self.dataset_params["list_modalities"],
            imu="imu" in self.dataset_params["list_modalities"],
            return_tuple=True,
            target_frames_in_window=self.dataset_params["target_fps"],
            window_set=load_csv(path),
            class_dict=self.lable_dict,
        )

    def setup(self, stage: Optional[str] = None):

        # Initialize data
        if stage in (None, "fit"):
            print("TRAIN")
            self.train = self.get_dataset("training")

            print("VALIDATION")
            self.val = self.get_dataset("validation")

        if stage in (None, "test"):
            print("TEST")
            self.test = self.get_dataset("test")
            self.predict = self.test

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=False,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )


class UnsupEgo4dDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, pin_memory, drop_last, dataset_params):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.dataset_params = dataset_params

        self.prepare_data_per_node = False
        self._log_hyperparams = True

        self.collate_fn = lambda data: collate_wrapper(
            data, self.dataset_params["list_modalities"]
        )

        # Get splits ready
        self.filter_video_uids_train = Split(random_split=0, split="training")
        self.filter_video_uids_validation = Split(random_split=0, split="validation")
        self.filter_video_uids_test = Split(random_split=0, split="test")

    def get_dataset(
        self,
        split: str,
        video_uid_sample_rate: float = 1.0,
        window_sample_rate: float = 1.0,
        max_n_windows_per_video: Optional[int] = None,
        shuffle_windows=True,
    ) -> Ego4dDatasetUnsupervised:

        if split == "training":
            filter_video_uids_split = self.filter_video_uids_train

        elif split == "validation":
            filter_video_uids_split = self.filter_video_uids_validation

        elif split == "test":
            filter_video_uids_split = self.filter_video_uids_test

        if video_uid_sample_rate != 1.0:
            filter_video_uids_split.scale(video_uid_sample_rate)

        return Ego4dDatasetUnsupervised(
            window_sec=self.dataset_params["window_sec"],
            video="video" in self.dataset_params["list_modalities"],
            imu="imu" in self.dataset_params["list_modalities"],
            audio="audio" in self.dataset_params["list_modalities"],
            return_tuple=False,
            target_frames_in_window=self.dataset_params["target_fps"],
            filter_video_uids=filter_video_uids_split.filter,
            window_sample_rate=window_sample_rate,
            max_n_windows_per_video=max_n_windows_per_video,
            shuffle_windows=shuffle_windows,
        )

    def setup(self, stage: Optional[str] = None):

        # Initialize data
        if stage in (None, "fit"):
            print("TRAIN")
            self.train = self.get_dataset("training")

            print("VALIDATION")
            self.val = self.get_dataset("validation")

        if stage in (None, "test"):
            print("TEST")
            self.test = self.get_dataset("test")
            self.predict = self.test

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=True,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
