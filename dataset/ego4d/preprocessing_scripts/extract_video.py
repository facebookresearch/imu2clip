# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import os
import glob
import argparse
import ffmpeg

from tqdm import tqdm


def preprocess_videos(fps, size, video_dir, output_dir):
    aw = 0.5
    ah = 0.5
    fps = fps
    size = size

    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(glob.glob(f"{video_dir}/*.mp4")):
        name_clip = filename.split("/")[-1].replace(".mp4", "")
        try:
            # Initialize paths.
            processed_video_tmp_path = os.path.join(output_dir, name_clip + "-tmp.mp4")
            processed_video_path = os.path.join(output_dir, name_clip + ".mp4")
            raw_video_path = os.path.join(video_dir, name_clip + ".mp4")

            if not os.path.exists(processed_video_path):
                _, _ = (
                    ffmpeg.input(raw_video_path)
                    .filter("fps", fps)
                    .crop(
                        "(iw - min(iw,ih))*{}".format(aw),
                        "(ih - min(iw,ih))*{}".format(ah),
                        "min(iw,ih)",
                        "min(iw,ih)",
                    )
                    .filter("scale", size, size)
                    .output(processed_video_tmp_path)
                    .overwrite_output()
                    .run(capture_stdout=True, quiet=True, cmd="ffmpeg")
                )
                os.rename(processed_video_tmp_path, processed_video_path)
        except Exception as e:
            print(f"{e} processing {name_clip}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Downframe video to given fps and resize frames"
    )
    parser.add_argument(
        "-v",
        "--video_dir",
        required=True,
        help="Directory with video files",
        default="path_to_videos/full_scale",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Output dir for audio clips",
        default="path_to_preprocess/full_videos/processed_videos",
    )
    parser.add_argument("-f", "--fps", required=True, help="Target fps", default=10)
    parser.add_argument(
        "-s", "--size", required=True, help="Target frame size", default=224
    )

    args = parser.parse_args()

    preprocess_videos(args.fps, args.size, args.video_dir, args.output_dir)
