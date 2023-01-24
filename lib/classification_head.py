# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

import pytorch_lightning as pl


class Head(pl.LightningModule):
    """
    Add linear layer on top of an encoder.
    """

    def __init__(self, encoder, size_embeddings, n_classes, classification_head=None):
        """
        encoder: IMU or Vision encoder that convert everything into one embeddings.
        size_embeddings: embedding size.
        n_classes: number of classes.
        """
        super().__init__()
        self.name = "Head"
        self.encoder = encoder
        if classification_head is not None:
            self.head = classification_head
        else:
            self.head = nn.Linear(size_embeddings, n_classes)

    def forward(self, batch):
        """
        Forward function
        """
        return self.head(self.encoder(batch))


class ZeroShotClassification(pl.LightningModule):
    """
    zero shot activities classification
    """

    def __init__(self, encoder, text_encoder, label_texts):
        """
        encoder: IMU or Vision encoder that convert everything into one embeddings.
        text_encoder: CLIP text encoder.
        n_classes: number of classes.
        """
        super().__init__()
        self.label_texts = label_texts
        self.text_encoder = text_encoder
        self.encoder = encoder

    def forward(self, batch):
        """
        Forward function
        """
        # Calculate features
        image_features = self.encoder(batch)
        text_features = self.text_encoder.get_text_embeddings(self.label_texts)
        image_features /= image_features.norm(dim=-1, keepdim=True)  # batch, emb_size
        text_features /= text_features.norm(
            dim=-1, keepdim=True
        )  # num_classes, emb_size
        return image_features @ text_features.T
