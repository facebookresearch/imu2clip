# Data preprocessing 

Both the Ego4d and Aria data are available upon request at: 
- Download: [Ego4d](https://ego4d-data.org/). Please make sure you download all the IMU data.
- Download: [Aria](https://facebookresearch.github.io/Aria_data_tools/docs/pilotdata/pilotdata-index/). Currently, we do not support Aria in this repo. This is due to custum data extraction pipeline (i.e., [VRS](https://github.com/facebookresearch/vrs#getting-started)).

After this step you must have a folder with ego4d data inside. 

## Resize and Scale frames in ego4d videos
We preprocess the videos in ego4d to make sure all the videos has same frame size (224x224) and FPS (10FPS). To do that, check the preprocessing script folder and run ```proprocess.sh```. Please make sure to use the correct paths. 

## Convert IMU csv into npy arrays

We preprocess the imu in ego4d to make sure all the imu has same sampling rate (200hz). To that, go to the preprocessing script folder and run ```proprocess.sh```. Please make sure to use the correct paths. 
