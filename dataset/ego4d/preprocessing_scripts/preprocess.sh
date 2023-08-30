# Copyright (c) Meta Platforms, Inc. and affiliates.

# Directory paths
ROOT_DIR="$(cd "$(dirname "$0")/../../../" && pwd)"
BASE_DIR="$ROOT_DIR/checkpoint"
FULL_VIDEOS_DIR="$BASE_DIR/full_videos"
CLIPS_DIR="$BASE_DIR/clips"

# Create necessary directories for video preprocessing
mkdir -p $BASE_DIR
echo "Creating directories for video preprocessing..."

mkdir -p $FULL_VIDEOS_DIR/processed_video
mkdir -p $CLIPS_DIR/processed_video

echo "Extracting full-scale videos..."
python extract_video.py -v $ROOT_DIR/v1/full_scale/ -o $FULL_VIDEOS_DIR/processed_video -f 10 -s 224

echo "Extracting video clips..."
python extract_video.py -v $ROOT_DIR/v1/clips/ -o $CLIPS_DIR/processed_video -f 10 -s 224

# Create necessary directories for IMU preprocessing
echo "Creating directories for IMU preprocessing..."
mkdir -p $CLIPS_DIR/processed_imu
mkdir -p $FULL_VIDEOS_DIR/processed_imu

# Run the IMU extraction scripts
echo "Extracting IMU data for full videos..."
python extract_imu.py -v $ROOT_DIR/v1/imu -o $FULL_VIDEOS_DIR/processed_imu
echo "Extracting IMU data for clips..."
python extract_imu.py -v $ROOT_DIR/v1/imu -o $CLIPS_DIR/processed_imu

echo "Preprocessing completed successfully!"
