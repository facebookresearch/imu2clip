# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import json
import string
import random
import glob
import numpy as np
import torch


random.seed(1234)


class Ego4dNarration(torch.utils.data.Dataset):
    DATA_PATH = "../v1/"

    def __init__(self):

        print("Start Loading windows")
        self.file_path = glob.glob(f"{self.DATA_PATH}/imu/*.npy")
        self.file_path = [
            pth.split("/")[-1].replace(".npy", "") for pth in self.file_path
        ]
        print(f"Number of window loaded {len(self.file_path)}")

    def __len__(self):
        return len(self.file_path)

    def _load_json(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        return data

    def _load_npy(self, npy_path):
        with open(npy_path, "rb") as f:
            npy_arr = np.load(f)
        return npy_arr

    def __getitem__(self, idx):
        f_path = self.file_path[idx]
        dict_out = {}
        dict_out["imu"] = torch.Tensor(
            self._load_npy(f"{self.DATA_PATH}/imu/{f_path}.npy")
        )
        dict_out["narration"] = self._load_json(
            f"{self.DATA_PATH}/narration/{f_path}.json"
        )

        return dict_out


def collate_fn(data):
    input_tensor_IMU = []
    input_tensor_NARRATION = []
    for d in data:
        input_tensor_IMU.append(d["imu"])
        input_tensor_NARRATION.append(d["narration"])

    dict_output = {}
    dict_output["imu"] = torch.stack(input_tensor_IMU)
    dict_output["narration"] = input_tensor_NARRATION

    return dict_output


def collate_fn_sanitized(data):
    input_tensor_IMU = []
    input_tensor_NARRATION = []
    for d in data:
        input_tensor_IMU.append(d["imu"])

        # Take only the first narration instance
        narration = d["narration"][0]

        # Removes artifacts, e.g. '#C', punctuations.
        narration = (
            narration.replace("#C C ", "")
            .replace("#C", "")
            .replace("#O ", "")
            .replace("#unsure", "something")
            .strip()
            .strip(string.punctuation)
            .lower()[:128]
        )

        input_tensor_NARRATION.append(narration)

    dict_output = {}
    dict_output["imu"] = torch.stack(input_tensor_IMU)
    dict_output["narration"] = input_tensor_NARRATION

    return dict_output
