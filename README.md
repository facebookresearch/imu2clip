## IMU2CLIP

This is the code for [IMU2CLIP](https://arxiv.org/abs/2210.14395), a novel pre-training approach to align Inertial Measurement Unit (IMU) motion sensor recordings with video and text, by projecting them into the joint representation space of Contrastive Language-Image Pre-training (CLIP). The proposed approach allows IMU2CLIP to translate human motions (as measured by IMU sensors) into their corresponding textual descriptions and videos -- while preserving the transitivity across these modalities.
To show the efficacy of the model, we explore several new IMU-based applications that IMU2CLIP enables, such as motion-based media retrieval and natural language reasoning tasks with motion data. In addition, we show that IMU2CLIP can significantly improve the downstream performance when fine-tuned for each application (e.g. activity recognition), demonstrating the universal usage of IMU2CLIP as a new pre-trained resource.

# Installation

```
conda create -n imu2clip python=3.8
conda activate imu2clip
pip install pytorch_lightning
pip install torchaudio
pip install torchvision
pip install git+https://github.com/openai/CLIP.git
pip install opencv-python
pip install matplotlib
pip install ffmpeg-python
pip install pandas
```

After installing all the library, check the in ```dataset/ego4d/README.md``` for instruction on how to preprocess the ego4d data. 

# Experiments
**To run an example train loop**
```
python pretraining.py
```

**To run a pretrained model in downstream task**
```
python downstream.py
```

In the config folder, you can find details hyperparamters for training IMU2CLIP with different contrastive losses. 

# Citation

```
@article{moon2022imu2clip,
  title={IMU2CLIP: Multimodal Contrastive Learning for IMU Motion Sensors from Egocentric Videos and Text},
  author={Moon, Seungwhan and Madotto, Andrea and Lin, Zhaojiang and Dirafzoon, Alireza and Saraf, Aparajita and Bearman, Amy and Damavandi, Babak},
  journal={arXiv preprint arXiv:2210.14395},
  year={2022}
}
```
# License

The majority of IMU2CLIP is licensed under CC-BY-NC, however portions of the project are available under separate license terms: [PyTorchLigtning](https://github.com/Lightning-AI/lightning/blob/master/LICENSE) is licensed under the Apache 2.0 license and [CLIP](https://github.com/openai/CLIP/blob/main/LICENSE) is licensed under the MIT License.

See [LICENSE](LICENSE.md) for details.
