# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

from bisect import bisect_left
from collections import defaultdict
import math
import os
import csv
import json
from typing import Any, List, Optional
import cv2
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import torchvision.transforms as transforms

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

torchaudio.set_audio_backend("sox_io")
import torchvision.io as io

PATH_EGO_META = "../v1/ego4d.json"
PATH_ARIA_META = "/fsx/andreamad8/aria/aria.json"


def load_json(json_path: str):
    """
    Load a json file
    """
    with open(json_path, "r", encoding="utf-8") as f_name:
        data = json.load(f_name)
    return data


def save_json(json_path: str, data_obj: Any):
    """
    Save a json file
    """
    with open(json_path, "w", encoding="utf-8") as f_name:
        json.dump(data_obj, f_name, indent=4)


def load_csv(csv_path: str):
    """
    Load a CSV file
    """
    with open(csv_path, "r", encoding="utf-8") as f_name:
        reader = csv.DictReader(f_name)
        data = []
        for row in reader:
            data.append(row)
    return data


def get_ego4d_metadata(types: str = "clip"):
    """
    Get ego4d metadata
    """
    return {
        clip[f"{types}_uid"]: clip for clip in load_json(PATH_EGO_META)[f"{types}s"]
    }


def get_aria_metadata():
    """
    Get ego4d metadata
    """
    return {clip["name"].replace("/", "_"): clip for clip in load_json(PATH_ARIA_META)}


def modality_checker(meta_video: dict):
    """
    Give the video metadata return which modality is available
    """
    has_imu = meta_video["has_imu"]
    has_audio = (
        False if meta_video["video_metadata"]["audio_start_sec"] is None else True
    )
    return has_imu, has_audio


def check_window_signal(info_t, w_s, w_e):
    length = w_e - w_s
    frame_offset = int(w_s * info_t.sample_rate)
    num_frames = int(length * info_t.sample_rate)
    if frame_offset + num_frames > int(info_t.num_frames):
        return False
    else:
        return True


def verify_window(video, audio_info, imu_info, w_s, w_e):
    audio = True if audio_info else False
    imu = True if imu_info else False
    if video is True and audio is True and imu is True:
        if check_window_signal(audio_info, w_s, w_e) and check_window_signal(
            imu_info, w_s, w_e
        ):
            return True
    if video is True and audio is True and imu is False:
        if check_window_signal(audio_info, w_s, w_e):
            return True
    if video is True and audio is False and imu is True:
        if check_window_signal(imu_info, w_s, w_e):
            return True
    if video is False and audio is True and imu is True:
        if check_window_signal(audio_info, w_s, w_e) and check_window_signal(
            imu_info, w_s, w_e
        ):
            return True
    if video is True and audio is False and imu is False:
        return True
    if video is False and audio is True and imu is False:
        if check_window_signal(audio_info, w_s, w_e):
            return True
    if video is False and audio is False and imu is True:
        if check_window_signal(imu_info, w_s, w_e):
            return True
    if video is False and audio is False and imu is False:
        return False
    return False


def get_windows_in_clip(s_time: float, e_time: float, window_sec: float, stride: float):
    """
    Given start and end time, return windows of size window_sec.
    If stride!=window_sec, convolve with stride.
    """
    windows = []
    for window_start, window_end in zip(
        np.arange(s_time, e_time, stride),
        np.arange(
            s_time + window_sec,
            e_time,
            stride,
        ),
    ):
        windows.append([window_start, window_end])
    return windows


def index_narrations():
    narration_raw = load_json("/datasets01/ego4d_track2/v1/annotations/narration.json")

    narration_dict = defaultdict(list)
    summary_dict = defaultdict(list)
    avg_len = []
    for v_id, narr in narration_raw.items():
        narr_list = []
        summ_list = []
        if "narration_pass_1" in narr:
            narr_list += narr["narration_pass_1"]["narrations"]
            summ_list += narr["narration_pass_1"]["summaries"]
        if "narration_pass_2" in narr:
            narr_list += narr["narration_pass_2"]["narrations"]
            summ_list += narr["narration_pass_2"]["summaries"]

        if len(narr_list) > 0:
            narration_dict[v_id] = [
                (
                    float(n_t["timestamp_sec"]),
                    n_t["narration_text"],
                    n_t["annotation_uid"],
                    n_t["timestamp_frame"],
                )
                for n_t in narr_list
            ]
            avg_len.append(len(narration_dict[v_id]))
        else:
            narration_dict[v_id] = []
        if len(summ_list) > 0:
            summary_dict[v_id] = [
                (
                    float(s_t["start_sec"]),
                    float(s_t["end_sec"]),
                    s_t["summary_text"],
                )
                for s_t in summ_list
            ]
        else:
            summary_dict[v_id] = []
    # print(f"Number of Videos with narration {len(narration_dict)}")
    # print(f"Avg. narration length {np.mean(avg_len)}")
    # print(f"Number of Videos with summaries {len(summary_dict)}")
    return narration_dict, summary_dict


def downsample_video(
    frames: torch.Tensor = torch.zeros(3, 10, 224, 224), targer_frames: int = 5
):
    """
    Downsample video to target number of frame. For example from [3,10,224,224] to [3,5,224,224]
    """
    temporal_dim = 1
    num_frames_sampled = frames.size(temporal_dim)
    # -1 because index starts from 0. linspace includes both ends in the sampled list
    selected_frame_indices = torch.linspace(
        0, num_frames_sampled - 1, targer_frames
    ).long()
    return torch.index_select(frames, temporal_dim, selected_frame_indices)


def get_video_frames(
    video_fn: str,
    video_start_sec: float,
    video_end_sec: float,
    target_frames_in_window: int = 10,
):
    """
    Given a video return the frames between video_start_sec and video_end_sec
    """
    vframes, _, info = io.read_video(
        video_fn,
        start_pts=video_start_sec,
        end_pts=video_end_sec,
        pts_unit="sec",
    )

    vframes = vframes.permute(3, 0, 1, 2)
    # pad frames
    if target_frames_in_window != vframes.size(1):
        vframes = downsample_video(vframes, target_frames_in_window)
    vframes = vframes / 255.0
    vframes = torch.stack(
        [transforms.functional.rotate(frames, -90) for frames in vframes]
    )
    return {"frames": vframes, "meta": info}


def get_signal_info(signal_fn: str):
    return torchaudio.info(signal_fn)


def get_signal_frames(signal_fn: str, video_start_sec: float, video_end_sec: float):
    """
    Given a signal track return the frames between video_start_sec and video_end_sec
    """
    info_t = get_signal_info(signal_fn)

    length = video_end_sec - video_start_sec
    aframes, _ = torchaudio.load(
        signal_fn,
        normalize=True,
        frame_offset=int(video_start_sec * info_t.sample_rate),
        num_frames=int(length * info_t.sample_rate),
    )
    return {"signal": aframes, "meta": info_t}


def display_image_list(
    images: np.array,
    title: Optional[List[str]] = None,
    columns: Optional[int] = 5,
    width: Optional[int] = 20,
    height: Optional[int] = 8,
    max_images: Optional[int] = 20,
    label_font_size: Optional[int] = 10,
    save_path_img: str = "",
) -> None:
    """
    Util function to plot a set of images with, and save it into
    manifold. If the labels are provided, they will be added as
    title to each of the image.

    Args:
        images: (numpy.ndarray of shape (batch_size, color, hight, width)) - batch of
                images

        labels: (List[str], optional) —  List of strings to be used a title for each img.
        columns: (int, optional) — Number of columns in the grid. Raws are compute accordingly.
        width: (int, optional) — Figure width.
        height: (int, optional) — Figure height.
        max_images: (int, optional) — Maximum number of figure in the grid.
        label_font_size: (int, optional) - font size of the lable in the figure
        save_path_img: (str, ) - path to the manifold to save the figure.

    Example:

        >>> img = torch.rand(2, 3, 224, 224)
        >>> lab = ["a cat", "a dog"]
        >>> display_image_list(
                img,
                lab,
                save_path_img="path_name.png",
            )
    """
    plt.rcParams["axes.grid"] = False

    if len(images) > max_images:
        images = images[0:max_images, :, :, :]

    height = max(height, int(len(images) / columns) * height)
    plt.figure(figsize=(width, height))
    for i in range(len(images)):

        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        # plt.imshow(transforms.ToPILImage()(images[i]).convert("RGB"))
        plt.imshow(images[i])
        plt.axis("off")

        if title:
            plt.title(title, fontsize=label_font_size)

    with open(save_path_img, "wb") as f_name:
        plt.savefig(fname=f_name, dpi=400)
    plt.close()


def display_animation(frames, title, save_path_gif):
    fig, ax = plt.subplots()
    frames = [[ax.imshow(frames[i])] for i in range(len(frames))]
    plt.title(title)
    ani = animation.ArtistAnimation(fig, frames)
    ani.save(save_path_gif, writer="imagemagick")
    plt.close()


def display_animation_imu(frames, imu, title, save_path_gif):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.set_title(title)
    ax2.set_title("Acc.")
    ax3.set_title("Gyro.")
    frames = [[ax1.imshow(frames[i])] for i in range(len(frames))]
    ani = animation.ArtistAnimation(fig, frames)

    ax2.plot(imu[0].cpu().numpy(), color="red")
    ax2.plot(imu[1].cpu().numpy(), color="blue")
    ax2.plot(imu[2].cpu().numpy(), color="green")
    ax3.plot(imu[3].cpu().numpy(), color="red")
    ax3.plot(imu[4].cpu().numpy(), color="blue")
    ax3.plot(imu[5].cpu().numpy(), color="green")
    plt.tight_layout()
    ani.save(save_path_gif, writer="imagemagick")
    plt.close()
