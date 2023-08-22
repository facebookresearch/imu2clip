# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import random
import os

from typing import Optional
import torch
import pytorch_lightning as pl
from dataset.utils import load_json
from dataset.MMdataloader import MMdataset
from dataset.utils import (
    get_ego4d_metadata,
    modality_checker,
    get_aria_metadata,
)

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
        self.set = load_json(f"../../splits/{split}_{random_split}.json")

        if video_uid_sample_rate != 1.0:
            self.scale(video_uid_sample_rate)

    def scale(self, video_uid_sample_rate: float):
        # Typically for debugging purposes, etc.
        assert video_uid_sample_rate < 1.0 and video_uid_sample_rate > 0.0
        n_videos_to_return = max(int(len(self.set) * video_uid_sample_rate), 1)
        print(f"Reducing to {n_videos_to_return} videos ...")
        self.set = set(list(self.set)[:n_videos_to_return])

    def filter(self, video_uid):
        if video_uid in self.set:
            return True
        return False


class MMDataModule(pl.LightningDataModule):
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
        self.data_path = "../v1/full_videos"
        self.meta_video = get_ego4d_metadata("video")
        self.video = "video" in self.dataset_params["list_modalities"]
        self.imu = "imu" in self.dataset_params["list_modalities"]
        self.audio = "audio" in self.dataset_params["list_modalities"]

        video_uids = list(self.meta_video.keys())
        video_uids = [
            video_uid
            for video_uid in video_uids
            if self.check_modality_clip_uid(video_uid)
        ]

        video_uids.remove("374832bf-f977-4e8b-b0e0-2f2ea1e38b5d")

        self.train_uids = [
            v_uid for v_uid in video_uids if self.filter_video_uids_train.filter(v_uid)
        ]
        self.validation_uids = [
            v_uid
            for v_uid in video_uids
            if self.filter_video_uids_validation.filter(v_uid)
        ]
        self.test_uids = [
            v_uid for v_uid in video_uids if self.filter_video_uids_test.filter(v_uid)
        ]

        print(f"Num. Train videos {len(self.train_uids)}")
        print(f"Num. Valid videos {len(self.validation_uids)}")
        print(f"Num. Tests videos {len(self.test_uids)}")

    def check_modality_clip_uid(self, video_uid):
        """
        Check which modality is avalibale in the clip based on the request input
        """
        has_imu, has_audio = modality_checker(self.meta_video[video_uid])
        if self.imu and not has_imu:
            return False
        if self.audio and (
            not has_audio
            or not os.path.exists(
                os.path.join(self.data_path, f"processed_audios/{video_uid}.wav")
            )
        ):
            return False
        return True

    def dummy(self, x_placeholder, y_placeholder, z_placeholder):
        return None

    def get_dataset(
        self,
        split: str,
        video_uid_sample_rate: float = 1.0,
        window_sample_rate: float = 1.0,
        max_n_windows_per_video: Optional[int] = None,
    ) -> MMdataset:

        if split == "training":
            video_uids = self.train_uids

        elif split == "validation":
            video_uids = self.validation_uids

        elif split == "test":
            video_uids = self.test_uids

        if video_uid_sample_rate != 1.0:
            # Typically for debugging purposes, etc.
            assert video_uid_sample_rate < 1.0 and video_uid_sample_rate > 0.0

            n_videos_to_return = max(int(len(video_uids) * video_uid_sample_rate), 1)
            print(f"Reducing to {n_videos_to_return} videos ...")
            video_uids = video_uids[:n_videos_to_return]

        return MMdataset(
            video_uids,
            video=self.video,
            audio=self.audio,
            imu=self.imu,
            return_tuple=False,
            window_sec=self.dataset_params["window_sec"],
            target_frames_in_window=self.dataset_params["target_fps"],
            dataset_name="ego4d_video",
            data_path=self.data_path,
            lable_fn=self.dummy,
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


class MMDataModuleARIA(pl.LightningDataModule):
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
        self.data_path = "/fsx/andreamad8/aria/"
        self.meta_video = get_aria_metadata()
        self.video = "video" in self.dataset_params["list_modalities"]
        self.imu = "imu" in self.dataset_params["list_modalities"]
        self.audio = "audio" in self.dataset_params["list_modalities"]

        video_uids = list(self.meta_video.keys())
        random.shuffle(video_uids)
        self.train_uids = video_uids[:100]
        self.validation_uids = video_uids[100:120]
        self.test_uids = video_uids[120:]

        print(f"Num. Train videos {len(self.train_uids)}")
        print(f"Num. Valid videos {len(self.validation_uids)}")
        print(f"Num. Tests videos {len(self.test_uids)}")

    def dummy(self, x_placeholder, y_placeholder, z_placeholder):
        return None

    def get_dataset(
        self,
        split: str,
        video_uid_sample_rate: float = 1.0,
        window_sample_rate: float = 1.0,
        max_n_windows_per_video: Optional[int] = None,
    ) -> MMdataset:

        if split == "training":
            video_uids = self.train_uids

        elif split == "validation":
            video_uids = self.validation_uids

        elif split == "test":
            video_uids = self.test_uids

        if video_uid_sample_rate != 1.0:
            # Typically for debugging purposes, etc.
            assert video_uid_sample_rate < 1.0 and video_uid_sample_rate > 0.0
            n_videos_to_return = max(int(len(video_uids) * video_uid_sample_rate), 1)
            print(f"Reducing to {n_videos_to_return} videos ...")
            video_uids = video_uids[:n_videos_to_return]

        return MMdataset(
            video_uids,
            video=self.video,
            audio=self.audio,
            imu=self.imu,
            return_tuple=False,
            window_sec=self.dataset_params["window_sec"],
            target_frames_in_window=self.dataset_params["target_fps"],
            dataset_name="aria",
            data_path=self.data_path,
            lable_fn=self.dummy,
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
