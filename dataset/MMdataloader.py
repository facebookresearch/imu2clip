# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import random
import copy
import os
from typing import Callable, List, Optional, Dict
from tqdm import tqdm
import torch
from dataset.utils import (
    get_ego4d_metadata,
    get_aria_metadata,
    get_video_frames,
    get_signal_frames,
    get_signal_info,
    get_windows_in_clip,
    verify_window,
)


class MMdataset(torch.utils.data.Dataset):
    """
    Datasets class for the multimodal Datasets
    This Datasets class loops over (Audio, Video, IMU) windows
    of n-sec.
    """

    def __init__(
        self,
        video_ids: List,
        video: bool = True,
        audio: bool = True,
        imu: bool = True,
        window_sec: float = 1.0,
        target_frames_in_window: int = 10,
        return_tuple: bool = True,
        return_label: bool = False,
        lable_fn: Callable[[int, int, str], str] = None,
        dataset_name: str = "ego4d_clip",
        data_path: str = "../v1/clips",
        window_sample_rate: float = 1.0,
        max_n_windows_per_video: Optional[int] = None,
    ):
        self.return_tuple = return_tuple
        self.window_sec = window_sec
        self.target_frames_in_window = target_frames_in_window
        if "ego4d_clip" == dataset_name:
            self.meta = get_ego4d_metadata("clip")
            self.duration_fn = lambda x: x["clip_metadata"]["mp4_duration_sec"]
        elif "ego4d_video" == dataset_name:
            self.meta = get_ego4d_metadata("video")
            self.duration_fn = lambda x: x["duration_sec"]
        elif "aria" == dataset_name:
            self.meta = get_aria_metadata()
            self.duration_fn = lambda x: x["video"]["duration"]

        self.data_path = data_path
        self.video = video
        self.audio = audio
        self.imu = imu
        self.return_label = return_label

        self.window_idx = []
        for video_uid in tqdm(video_ids):

            video_duration = self.duration_fn(self.meta[video_uid])
            audio_info, imu_info = None, None
            if self.audio:
                audio_info = get_signal_info(
                    os.path.join(self.data_path, f"processed_audios/{video_uid}.wav")
                )
            if self.imu:
                imu_info = get_signal_info(
                    os.path.join(self.data_path, f"processed_imu/{video_uid}.wav"),
                )

            windows_in_clip = get_windows_in_clip(
                s_time=0,
                e_time=video_duration,
                window_sec=window_sec,
                stride=window_sec,
            )

            n_windows_per_video = 0
            if max_n_windows_per_video is not None:
                # e.g. for more balanced sampling of windows s.t.
                # a long clip does not dominate the data
                random.shuffle(windows_in_clip)

            for (w_s, w_e) in windows_in_clip:
                if verify_window(video, audio_info, imu_info, w_s, w_e):
                    input_dict = {
                        "window_start": w_s,
                        "window_end": w_e,
                        "video_uid": video_uid,
                        "label": lable_fn(w_s, w_e, video_uid),
                    }

                    if (
                        max_n_windows_per_video is not None
                        and n_windows_per_video >= max_n_windows_per_video
                    ):
                        continue
                    if (
                        window_sample_rate != 1.0
                        and random.random() > window_sample_rate
                    ):
                        continue

                    self.window_idx.append(input_dict)
                    n_windows_per_video += 1

        print(f"There are {len(self.window_idx)} windows to process.")

    def __len__(self):
        return len(self.window_idx)

    def __getitem__(self, idx):
        dict_out = copy.deepcopy(self.window_idx[idx])
        uid = dict_out["video_uid"]
        w_s = dict_out["window_start"]
        w_e = dict_out["window_end"]

        if self.video:
            dict_out["video"] = get_video_frames(
                video_fn=os.path.join(self.data_path, f"processed_videos/{uid}.mp4"),
                video_start_sec=w_s,
                video_end_sec=w_e,
                target_frames_in_window=self.target_frames_in_window,
            )

        if self.audio:
            dict_out["audio"] = get_signal_frames(
                signal_fn=os.path.join(self.data_path, f"processed_audios/{uid}.wav"),
                video_start_sec=w_s,
                video_end_sec=w_e,
            )

        if self.imu:
            dict_out["imu"] = get_signal_frames(
                signal_fn=os.path.join(self.data_path, f"processed_imu/{uid}.wav"),
                video_start_sec=w_s,
                video_end_sec=w_e,
            )

        if self.return_tuple:
            tuple_out = ()

            tuple_out = tuple_out + ((dict_out["video"]["frames"],))
            if self.audio:
                tuple_out = tuple_out + ((dict_out["audio"]["signal"],))
            if self.imu:
                tuple_out = tuple_out + ((dict_out["imu"]["signal"],))
            if self.return_label:
                tuple_out = tuple_out + (dict_out["label"],)

            tuple_out = tuple_out + (uid,)

            return tuple_out

        return dict_out


class Ego4dDatasetSupervised(torch.utils.data.Dataset):
    """
    Datasets class for the ego4d Dataset
    This dataloder loops over (Audio, Video, IMU) windows
    of n-sec with labels
    """

    def __init__(
        self,
        video=False,
        audio=False,
        imu=False,
        window_sec: float = 1.0,
        target_frames_in_window: int = 10,
        return_tuple: bool = True,
        cache_imu: bool = False,
        window_set: List = [],
        class_dict: Dict = {},
    ):
        self.return_tuple = return_tuple
        self.target_frames_in_window = target_frames_in_window

        self.meta_video = get_ego4d_metadata("video")
        self.video = video
        self.audio = audio
        self.imu = imu
        self.class_dict = class_dict
        self.data_path = "../v1/full_videos"

        self.window_idx = window_set
        print(f"There are {len(self.window_idx)} windows to process.")

    def __len__(self):
        return len(self.window_idx)

    def __getitem__(self, idx):
        dict_out = copy.deepcopy(self.window_idx[idx])
        uid = dict_out["video_uid"]
        w_s = int(dict_out["window_start"])
        w_e = int(dict_out["window_end"])
        text = dict_out["label"]

        if self.video:
            dict_out["video"] = get_video_frames(
                video_fn=os.path.join(self.data_path, f"processed_videos/{uid}.mp4"),
                video_start_sec=w_s,
                video_end_sec=w_e,
                target_frames_in_window=self.target_frames_in_window,
            )

        if self.audio:
            dict_out["audio"] = get_signal_frames(
                signal_fn=os.path.join(self.data_path, f"processed_audios/{uid}.wav"),
                video_start_sec=w_s,
                video_end_sec=w_e,
            )

        if self.imu:
            dict_out["imu"] = get_signal_frames(
                signal_fn=os.path.join(self.data_path, f"processed_imu/{uid}.wav"),
                video_start_sec=w_s,
                video_end_sec=w_e,
            )
            if dict_out["imu"]["signal"].size(1) < 200:
                dict_out["imu"]["signal"] = torch.zeros(6, 1000)

        dict_out["label"] = self.class_dict[text]

        if self.return_tuple:
            tuple_out = ()

            if self.video:
                tuple_out = tuple_out + (dict_out["video"]["frames"].float(),)
            if self.audio:
                tuple_out = tuple_out + (dict_out["audio"]["signal"].float(),)
            if self.imu:
                tuple_out = tuple_out + (dict_out["imu"]["signal"].float(),)
            tuple_out = tuple_out + (self.class_dict[text],)

            return tuple_out

        return dict_out
