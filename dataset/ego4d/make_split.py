# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import random
from utils.utils import (
    index_narrations,
    modality_checker,
    get_ego4d_metadata,
    save_json,
)


def get_narration_number(video_set_uid, narration_dict):
    num_narration = []
    for video_uid, narrations in narration_dict.items():
        if video_uid in video_set_uid:
            num_narration.append(len(narrations))
    return sum(num_narration)


def get_video_uid_imu(narration_dict, meta_video):
    video_with_imu = []
    for video_uid, _ in narration_dict.items():
        has_imu, _ = modality_checker(meta_video[video_uid])
        if has_imu:
            video_with_imu.append(video_uid)
    return video_with_imu


narration_dict, _ = index_narrations()
meta_video = get_ego4d_metadata("video")

video_with_imu = get_video_uid_imu(narration_dict, meta_video)

for i in range(10):
    print(f"Random shuffle {i}")
    random.shuffle(video_with_imu)

    train_samples = int(len(video_with_imu) * 0.70)

    dataset_set = video_with_imu[:train_samples]
    test_set = video_with_imu[train_samples:]

    train_samples = int(len(dataset_set) * 0.90)
    training_set = dataset_set[:train_samples]
    validation_set = dataset_set[train_samples:]

    print(f"Video in Training: {len(training_set)}")
    print(
        f"Narration in Training: {get_narration_number(training_set, narration_dict)}"
    )
    print()
    print(f"Video in Validation: {len(validation_set)}")
    print(
        f"Narration in Validation: {get_narration_number(validation_set, narration_dict)}"
    )
    print()
    print(f"Video in Testing: {len(test_set)}")
    print(f"Narration in Testing: {get_narration_number(test_set, narration_dict)}")
    print()
    print()
    print()

    save_json(f"../../splits/training_{i}.json", training_set)
    save_json(f"../../splits/validation_{i}.json", validation_set)
    save_json(f"../../splits/test_{i}.json", test_set)
