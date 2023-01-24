# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from matplotlib import pyplot as plt

def display_imu(imu, imu_rs):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1)

    ax1.set_title("Acc.")
    ax2.set_title("Gyro.")
    ax1.plot(imu[0], color="red")
    ax1.plot(imu[1], color="blue")
    ax1.plot(imu[2], color="green")
    ax2.plot(imu[3], color="red")
    ax2.plot(imu[4], color="blue")
    ax2.plot(imu[5], color="green")

    ax3.set_title("Acc.")
    ax4.set_title("Gyro.")
    ax3.plot(imu_rs[0], color="red")
    ax3.plot(imu_rs[1], color="blue")
    ax3.plot(imu_rs[2], color="green")
    ax4.plot(imu_rs[3], color="red")
    ax4.plot(imu_rs[4], color="blue")
    ax4.plot(imu_rs[5], color="green")

    plt.tight_layout()
    plt.savefig("imu.png", dpi=300)
    plt.close()

def resample(
    signals: np.ndarray,
    timestamps: np.ndarray,
    original_sample_rate: int,
    resample_rate: int,
):
    """
    Resamples data to new sample rate
    """
    signals = torch.as_tensor(signals)
    timestamps = torch.from_numpy(timestamps).unsqueeze(-1)
    signals = torchaudio.functional.resample(
        waveform=signals.data, orig_freq=original_sample_rate, new_freq=resample_rate,
    ).numpy()

    nsamples = len(signals.T)

    period = 1 / resample_rate

    # timestamps are expected to be shape (N, 1)
    initital_seconds = timestamps[0] / 1e3

    ntimes = (torch.arange(nsamples) * period).view(-1, 1) + initital_seconds

    timestamps = (ntimes * 1e3).squeeze().numpy()
    return signals, timestamps


def resampleIMU(signal, timestamps):
    sampling_rate = int(1000 * (1 / (np.mean(np.diff(timestamps)))))
    print(f"SR: {sampling_rate}")
    # resample all to 200hz
    if sampling_rate != 200:
        signal, timestamps = resample(signal, timestamps, sampling_rate, 200)
        sampling_rate = int(1000 * (1 / (np.mean(np.diff(timestamps)))))
    print(f"SR AFTER: {sampling_rate}")
    return signal, timestamps


def load_imu(imu_csv_path, duration):
    df = pd.read_csv(imu_csv_path)

    # delete row with NaN timestamps
    df = df.dropna(subset=["canonical_timestamp_ms"])

    # sort by canonical_timestamp_ms
    df = df.sort_values(by=["canonical_timestamp_ms"])

    accl_x = df["accl_x"].fillna(0).tolist()
    accl_y = df["accl_y"].fillna(0).tolist()
    accl_z = df["accl_z"].fillna(0).tolist()
    gyro_x = df["gyro_x"].fillna(0).tolist()
    gyro_y = df["gyro_y"].fillna(0).tolist()
    gyro_z = df["gyro_z"].fillna(0).tolist()
    timestamps = df["canonical_timestamp_ms"].to_numpy()

    signal = np.array([accl_x, accl_y, accl_z, gyro_x, gyro_y, gyro_z,])
    # print(f"Video duration {duration}s")
    # print(f"IMU duration {(timestamps[-1]-timestamps[0])/1000}s")

    # print(signal.shape)
    signal_rs, timestamps_rs = resampleIMU(signal, timestamps)
    # print(signal_rs.shape)
    # print(f"IMU resample duration {(timestamps_rs[-1]-timestamps_rs[0])/1000}s")
    # display_imu(signal, signal_rs)
    # input()
    return signal_rs, timestamps_rs

def load_json(json_path: str):
    """
    Load a json file
    """
    with open(json_path, "r", encoding="utf-8") as f_name:
        data = json.load(f_name)
    return data

def get_ego4d_metadata(types: str = "clip"):
    return {
        clip[f"{types}_uid"]: clip
        for clip in load_json("../ego4d.json")[f"{types}s"]
    }


def process_imu_files(video_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    meta_video = get_ego4d_metadata("video")
    for filename in tqdm(glob.glob(f"{video_dir}/*.csv")):
        name_clip = filename.split("/")[-1].replace(".csv", "")
        duration = meta_video[name_clip]["video_metadata"][
                "video_duration_sec"
            ]
        signal, timestamps = load_imu(filename, duration)

        with open(f"{output_dir}/{name_clip}.npy", "wb") as file:
            np.save(file, signal)
        with open(f"{output_dir}/{name_clip}_timestamps.npy", "wb") as file:
            np.save(file, timestamps)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract npy from the given csv imu files")
    parser.add_argument(
        "-v",
        "--video_dir",
        required=False,
        help="Directory with imu csv files",
        default="path_to_imu/imu",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Output dir for imu files",
        default="path_to_preprocess/full_videos/processed_imu",
    )

    args = parser.parse_args()

    process_imu_files(args.video_dir, args.output_dir)
