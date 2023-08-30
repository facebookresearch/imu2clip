# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import copy
import json
import math
import random
import os
import numpy as np
from tqdm import tqdm
import torch
from dataset.ego4d.utils.utils import (
    get_ego4d_metadata,
    modality_checker,
    get_video_frames,
    get_audio_frames,
    get_imu_frames,
    get_windows_in_clip,
    load_json,
)
from typing import Callable, Optional

random.seed(1234)
DATA_PATH = "../checkpoint/full_videos"


class Ego4dDatasetUnsupervised(torch.utils.data.Dataset):
    """
    Datasets class for the ego4d Dataset
    This dataloder loops over (Audio, Video, IMU) windows
    of n-sec.
    """

    def __init__(
        self,
        video=True,
        audio=True,
        imu=True,
        window_sec: float = 1.0,
        target_frames_in_window: int = 10,
        return_tuple: bool = True,
        cache_imu: bool = True,
        filter_video_uids: Callable[[str], bool] = lambda x: True,
        window_sample_rate: float = 1.0,
        max_n_windows_per_video: Optional[int] = None,
        shuffle_windows: bool = True,
    ):
        self.return_tuple = return_tuple
        self.cache_imu = {"cache": cache_imu, "path": "../../tmp/video_imu"}
        if cache_imu and not os.path.exists(self.cache_imu["path"]):
            os.makedirs(self.cache_imu["path"], exist_ok=True)
        self.window_sec = window_sec
        self.target_frames_in_window = target_frames_in_window
        self.meta_video = get_ego4d_metadata("video")
        self.video = video
        self.audio = audio
        self.imu = imu
        bad_imus = []
        if window_sec == 5.0:
            path_bad_imu_json = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "bad_imu_windows_5.0.json"
            )
            bad_imus = load_json(path_bad_imu_json)

            bad_imus = set([f"{uid}_{start}_{end}" for uid, start, end in bad_imus])

        self.window_idx = []
        for video_uid in tqdm(self.meta_video.keys()):
            if not filter_video_uids(video_uid):
                continue
            if not self.check_modality_clip_uid(video_uid):
                continue
            video_duration = self.meta_video[video_uid]["video_metadata"][
                "video_duration_sec"
            ]

            windows_in_clip = get_windows_in_clip(
                s_time=0,
                e_time=video_duration,
                window_sec=window_sec,
                stride=window_sec,
            )
            n_windows_per_video = 0
            if max_n_windows_per_video is not None and shuffle_windows:
                # e.g. for more balanced sampling of windows s.t.
                # a long clip does not dominate the data
                random.shuffle(windows_in_clip)

            for (w_s, w_e) in windows_in_clip:
                if f"{video_uid}_{w_s}_{w_e}" in bad_imus:
                    continue

                input_dict = {
                    "window_start": w_s,
                    "window_end": w_e,
                    "video_uid": video_uid,
                }
                if (
                    max_n_windows_per_video is not None
                    and n_windows_per_video >= max_n_windows_per_video
                ):
                    continue
                if window_sample_rate != 1.0 and random.random() > window_sample_rate:
                    continue

                self.window_idx.append(input_dict)
                n_windows_per_video += 1

        print(f"There are {len(self.window_idx)} windows to process.")

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
                os.path.join(DATA_PATH, f"processed_audios/{video_uid}.wav")
            )
        ):
            return False
        return True

    def __len__(self):
        return len(self.window_idx)

    def __getitem__(self, idx):
        dict_out = copy.deepcopy(self.window_idx[idx])
        uid = dict_out["video_uid"]
        w_s = dict_out["window_start"]
        w_e = dict_out["window_end"]
        tuple_out = ()

        if self.video:
            dict_out["video"] = get_video_frames(
                video_fn=os.path.join(DATA_PATH, f"processed_videos/{uid}.mp4"),
                video_start_sec=w_s,
                video_end_sec=w_e,
                target_frames_in_window=self.target_frames_in_window,
            )
            tuple_out = tuple_out + (dict_out["video"]["frames"],)

        if self.audio:
            dict_out["audio"] = get_audio_frames(
                audio_fn=os.path.join(DATA_PATH, f"processed_audios/{uid}.wav"),
                video_start_sec=w_s,
                video_end_sec=w_e,
            )
            tuple_out = tuple_out + (dict_out["audio"]["signal"],)

        if self.imu:
            dict_out["imu"] = get_imu_frames(
                uid=uid,
                video_start_sec=w_s,
                video_end_sec=w_e,
                cache=self.cache_imu,
            )
            tuple_out = tuple_out + (dict_out["imu"]["signal"],)

        if self.return_tuple:
            return tuple_out

        return dict_out


def collate_fn(data):
    input_tensor_IMU = []
    input_tensor_NARRATION = []
    for d in data:
        input_tensor_IMU.append(d["imu"]["signal"])
        input_tensor_NARRATION.append(d["narration"])

    dict_output = {}
    dict_output["imu"] = torch.stack(input_tensor_IMU).float()
    dict_output["narration"] = input_tensor_NARRATION

    return dict_output


def collate_fn_video(data):
    input_tensor_IMU = []
    input_tensor_video = []
    # input_tensor_NARRATION = []
    for d in data:
        input_tensor_IMU.append(d["imu"]["signal"])
        # input_tensor_NARRATION.append(d["narration"])
        input_tensor_video.append(d["video"]["frames"])

    dict_output = {}
    dict_output["video"] = torch.stack(input_tensor_video).float()
    dict_output["imu"] = torch.stack(input_tensor_IMU).float()
    # dict_output["narration"] = input_tensor_NARRATION

    return dict_output
