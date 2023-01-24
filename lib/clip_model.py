# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

from typing import List, Optional
import numpy as np
import clip
import torch
import json
from PIL import Image
import pytorch_lightning as pl


class ClipPLModel(pl.LightningModule):

    # Define some arbitrary classes
    DEFFAULT_LABELS = [
        "botanical_garden",
        "bow_window/indoor",
        "bowling_alley",
        "boxing_ring",
        "bridge",
        "building_facade",
        "bullring",
        "burial_chamber",
        "bus_interior",
        "bus_station/indoor",
        "butchers_shop",
        "butte",
        "cabin/outdoor",
        "cafeteria",
        "campsite",
        "campus",
    ]

    def __init__(self, *args, **kwargs):
        super(ClipPLModel, self).__init__()
        print("Loading clip model ...")
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        self.flag_freeze = kwargs.pop("freeze", True)
        self.video_encoder_name = kwargs.pop("video_encoder_name", "clip_1frame")

        if self.flag_freeze:
            self.eval()
            self.freeze()

    def forward(
        self, img: np.array, labels: Optional[List] = None, device: Optional[str] = None
    ):

        # Load optional values
        labels = self.DEFFAULT_LABELS if labels is None else labels
        device = self.device if device is None else device

        # Compute image features
        image = self.preprocess(img).unsqueeze(0).to(device)
        text = clip.tokenize(labels).to(device)

        # Compute similarity with labels
        logits_per_image, logits_per_text = self.clip_model(image, text)
        probs = logits_per_image.softmax(dim=-1)
        return probs, labels

    def get_text_embeddings(self, text: List[str], device: Optional[str] = None):

        device = self.device if device is None else device

        # Compute text features
        text_tokens = clip.tokenize(text).to(device)
        text_features = self.clip_model.encode_text(text_tokens)

        return text_features

    def get_img_embeddings(self, img, device: Optional[str] = None):
        # img: [batch_size x 3 x grid x grid]

        # Compute image features
        device = self.device if device is None else device
        # img = self.preprocess(img).unsqueeze(0).to(device)
        img_features = self.clip_model.encode_image(img)

        return img_features

    def get_video_embeddings(self, video, device: Optional[str] = None):
        # Take the first frame of the video input, and encode as an image
        # TODO: implement an actual temporal video encoder, or Clip4CLIP, etc.
        # video: [batch_size x 3 x n_frames x grid x grid]
        device = self.device if device is None else device

        # Compute video features
        if self.video_encoder_name == "clip_1frame":
            mid_frame_index = int(video.shape[2] / 2)
            frame = video[:, :, mid_frame_index, :, :]
            video_features = self.get_img_embeddings(frame)

        elif self.video_encoder_name == "clip_avg_frames":
            # For speed purposes, we just use 3 frames
            start_frame_index = 0
            mid_frame_index = int(video.shape[2] / 2)
            last_frame_index = -1

            video_features = self.get_img_embeddings(
                video[:, :, start_frame_index, :, :]
            )
            video_features += self.get_img_embeddings(
                video[:, :, mid_frame_index, :, :]
            )
            video_features += self.get_img_embeddings(
                video[:, :, last_frame_index, :, :]
            )

        return video_features

    def classify_from_clip_embeddings(
        self,
        input_clip_embeddings,
        labels: Optional[List] = None,
        device: Optional[str] = None,
        top_k: Optional[int] = 2,
    ) -> List[str]:

        # Load optional values
        labels = self.DEFFAULT_LABELS if labels is None else labels
        device = self.device if device is None else device

        # Compute text embeddings of the classes
        label_features = self.get_text_embeddings(labels, device)

        #  Compute similarity
        similarities = torch.nn.CosineSimilarity(dim=1)(
            label_features, input_clip_embeddings.to(device)
        )

        return get_top_k_predictions(similarities, labels, top_k)

    def classify_image(
        self,
        img: np.array,
        labels: Optional[List] = None,
        device: Optional[str] = None,
        top_k: Optional[int] = 2,
    ) -> List[str]:

        # Get softmax probs
        probs, labels = self(img, labels, device)

        # Get the label predictions
        return get_top_k_predictions(probs, labels, top_k)

    def get_img_file_from_path(self, path_img: str):
        return Image.open(path_img)

    def classify_image_from_path(
        self,
        path_image: str,
        labels: Optional[List] = None,
        device: Optional[str] = None,
        top_k: Optional[int] = 2,
    ) -> List[str]:
        image = self.get_img_file_from_path(path_image)
        return self.classify(image, labels=labels, device=device, top_k=top_k)


def get_top_k_predictions(probs, labels: List, top_k: Optional[int] = 2) -> List[str]:

    # probs: 1 x n_classes
    # Get the label predictions
    pred_classes = probs.topk(k=top_k).indices.cpu().tolist()[0]
    pred_class_names = [labels[int(i)] for i in pred_classes]
    return pred_class_names
