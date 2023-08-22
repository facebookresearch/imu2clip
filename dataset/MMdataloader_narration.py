# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import copy
import json
import math
import random
import os
import string
import numpy as np
from tqdm import tqdm
import torch
from dataset.utils import (
    get_ego4d_metadata,
    load_json,
    modality_checker,
    get_video_frames,
    get_signal_frames,
    index_narrations,
    verify_window,
    get_signal_info,
)
from typing import Callable, Dict, List, Optional

random.seed(1234)
DATA_PATH = "../checkpoint/full_videos"


class Ego4dDataset(torch.utils.data.Dataset):
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
        narr=False,
        window_sec: float = 1.0,
        target_frames_in_window: int = 10,
        return_tuple: bool = True,
        cache_imu: bool = False,
        clean_narration_func: Callable[[str], str] = lambda x: x,
        filter_narration_func: Callable[[str], bool] = lambda x: True,
        filter_video_uids: Callable[[str], bool] = lambda x: True,
        window_sample_rate: float = 1.0,
        max_n_windows_per_video: Optional[int] = None,
    ):
        self.return_tuple = return_tuple
        self.window_sec = window_sec
        self.target_frames_in_window = target_frames_in_window

        self.meta_video = get_ego4d_metadata("video")
        self.video = video
        self.audio = audio
        self.imu = imu
        self.narr = narr
        # self.cache = cache

        narration_dict, _ = index_narrations()

        self.window_idx = []
        for video_uid, narrations in tqdm(narration_dict.items()):
            if video_uid == "374832bf-f977-4e8b-b0e0-2f2ea1e38b5d":
                continue
            if not filter_video_uids(video_uid):
                continue
            if not self.check_modality_clip_uid(video_uid):
                continue
            video_duration = self.meta_video[video_uid]["video_metadata"][
                "video_duration_sec"
            ]
            n_windows_per_video = 0

            if max_n_windows_per_video is not None:
                random.shuffle(narrations)

            audio_info, imu_info = None, None
            if self.audio:
                audio_info = get_signal_info(
                    os.path.join(DATA_PATH, f"processed_audios/{video_uid}.wav")
                )
            if self.imu:
                imu_info = get_signal_info(
                    os.path.join(DATA_PATH, f"processed_imu/{video_uid}.wav"),
                )
            for (timestamp, text, a_uid, _) in narrations:
                if not filter_narration_func(text):
                    continue
                if (
                    max_n_windows_per_video is not None
                    and n_windows_per_video >= max_n_windows_per_video
                ):
                    continue
                if window_sample_rate != 1.0 and random.random() > window_sample_rate:
                    continue

                # check if it's the timestamp is at the very beginning
                if timestamp <= window_sec * 2:
                    w_s = 0.0
                    w_e = window_sec * 2
                # check if it's the time stamp is at the very end
                elif timestamp + window_sec * 2 >= video_duration:
                    w_s = video_duration - window_sec * 2
                    w_e = video_duration
                # else get a window of data around the timestamps
                else:
                    w_s = timestamp - window_sec
                    w_e = timestamp + window_sec

                w_s = int(math.floor(w_s))
                w_e = int(math.floor(w_e))
                try:
                    assert w_e - w_s == window_sec * 2
                except AssertionError:
                    continue

                if verify_window(self.video, audio_info, imu_info, w_s, w_e):
                    input_dict = {
                        "window_start": w_s,
                        "window_end": w_e,
                        "video_uid": video_uid,
                        "narration_uid": a_uid,
                        "text": clean_narration_func(text),
                    }
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
        text = dict_out["text"]

        if self.video:
            dict_out["video"] = get_video_frames(
                video_fn=os.path.join(DATA_PATH, f"processed_videos/{uid}.mp4"),
                video_start_sec=w_s,
                video_end_sec=w_e,
                target_frames_in_window=self.target_frames_in_window,
            )

        if self.audio:
            dict_out["audio"] = get_signal_frames(
                signal_fn=os.path.join(DATA_PATH, f"processed_audios/{uid}.wav"),
                video_start_sec=w_s,
                video_end_sec=w_e,
            )

        if self.imu:
            dict_out["imu"] = get_signal_frames(
                signal_fn=os.path.join(DATA_PATH, f"processed_imu/{uid}.wav"),
                video_start_sec=w_s,
                video_end_sec=w_e,
            )

        if self.narr:
            dict_out["narration"] = text

        if self.return_tuple:
            tuple_out = ()

            if self.video:
                tuple_out = tuple_out + (dict_out["video"]["frames"],)
            if self.audio:
                tuple_out = tuple_out + (dict_out["audio"]["signal"],)
            if self.imu:
                tuple_out = tuple_out + (dict_out["imu"]["signal"],)
            if self.narr:
                tuple_out = tuple_out + (text,)

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


def collate_fn_video_text(data):
    # TODO: Need to refactor
    input_tensor_IMU = []
    input_tensor_video = []
    input_tensor_NARRATION = []

    for d in data:
        input_tensor_IMU.append(d["imu"]["signal"])
        input_tensor_NARRATION.append(d["narration"])
        input_tensor_video.append(d["video"]["frames"])

    dict_output = {}
    dict_output["video"] = torch.stack(input_tensor_video).float()
    dict_output["imu"] = torch.stack(input_tensor_IMU).float()
    dict_output["narration"] = input_tensor_NARRATION

    assert len(dict_output["video"]) == len(dict_output["imu"])
    assert len(dict_output["video"]) == len(dict_output["narration"])

    return dict_output


def filter_narration(narration_text: str) -> bool:
    if "#c" in narration_text.lower():
        return True
    return False


def clean_narration_text(narration_text: str) -> str:
    return (
        narration_text.replace("#C C ", "")
        .replace("#C", "")
        .replace("#unsure", "something")
        .strip()
        .strip(string.punctuation)
        .lower()[:128]
    )
