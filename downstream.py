# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.

import random
import json
from datetime import datetime
import torch
import pytorch_lightning as pl
from lib.imu_models import MW2StackRNNPooling
from lib.classification_head import Head, ZeroShotClassification
from lib.clip_model import ClipPLModel
from lib.train_modules import ClassificationModule
from lib.data_modules import SupervisedEgo4dDataModule
from argparse import ArgumentParser
import yaml


def train_downstream(configs):

    random.seed(1234)

    # Load Model Parameters
    model_hparams = configs.get("model_hparams", {})
    model_name = model_hparams.get("model_name")
    model_suffix = model_hparams.get("model_suffix", "")
    imu_encoder_name = model_hparams.get("imu_encoder_name")
    window_sec = model_hparams.get("window_sec")
    target_fps = model_hparams.get("target_fps")

    # Params for the trainer
    train_hparams = configs.get("train_hparams", {})
    list_modalities = train_hparams.get("list_modalities")
    limit_train_batches = train_hparams.get("limit_train_batches")
    batch_size = train_hparams.get("batch_size")
    max_epochs = train_hparams.get("max_epochs")
    gpus = train_hparams.get("gpus")
    num_workers_for_dm = train_hparams.get("num_workers_for_dm")
    test_only = train_hparams.get("test_only")
    zero_shot = train_hparams.get("zero_shot")
    trainer_strategy = train_hparams.get("trainer_strategy")
    freeze_modality = train_hparams.get("freeze_modality")
    path_load_pretrained_imu_encoder = train_hparams.get(
        "path_load_pretrained_imu_encoder"
    )

    # Paths, etc.
    path_root_save_dir = f"./saved/{model_name}"
    list_modalities.sort()
    str_modality_initials = "".join([m[0] for m in list_modalities])
    model_identifier = (
        f"{model_name}_{str_modality_initials}_ie_{imu_encoder_name}_w_{window_sec}"
    )
    if model_suffix != "":
        model_identifier += "_" + model_suffix
    else:
        model_identifier += "_%d" % (int(datetime.now().timestamp() % 10000))
    path_save_checkpoint = f"{path_root_save_dir}/{model_identifier}.ckpt"
    result_path = f"./results/{model_identifier}"

    # Initialize the data module
    dataset_params = {
        "window_sec": window_sec,
        "target_fps": target_fps,
        "list_modalities": list_modalities,
    }

    datamodule = SupervisedEgo4dDataModule(
        batch_size=batch_size,
        num_workers=num_workers_for_dm,
        pin_memory=True,
        drop_last=True,
        dataset_params=dataset_params,
    )

    # get embeddings from label texts
    text_encoder = ClipPLModel(freeze=True)
    label_texts = list(datamodule.lable_dict.keys())

    encoder = MW2StackRNNPooling(size_embeddings=512)

    if path_load_pretrained_imu_encoder:
        # Load the parameters
        encoder.load_state_dict(torch.load(path_load_pretrained_imu_encoder))
        print("loaded pretrained imu model")

    if freeze_modality:
        encoder.eval()
        encoder.freeze()

    if zero_shot:
        # Initialize the training module for the classification model
        model = ClassificationModule(
            model=ZeroShotClassification(
                encoder=encoder, text_encoder=text_encoder, label_texts=label_texts
            )
        )
    else:
        # Initialize the training module for the classification model
        model = ClassificationModule(
            model=Head(
                encoder=encoder, size_embeddings=512, n_classes=datamodule.n_classes
            )
        )
    # Checkpoint settings
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=path_root_save_dir,
        filename=f"{model_identifier}" + "-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=gpus,
        limit_train_batches=limit_train_batches,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )

    if not test_only:
        # Start training
        print("Start training ...")
        trainer.fit(model, datamodule=datamodule)

        # Save the checkpoint & encoder to a temp folder
        print("Saving the checkpoint ...")
        trainer.save_checkpoint(path_save_checkpoint)

    else:
        print("Skipping training ...")

    print("Start testing ...")
    metrics = trainer.test(model, datamodule, ckpt_path=None if test_only else "best")
    result_path += f"_results.json"
    with open(result_path, "w") as f:
        json.dump({"metrics": metrics, "configs": configs}, f, indent=4)

    return metrics


if __name__ == "__main__":
    parser = ArgumentParser()

    # Main parameters are defined in a YAML file
    parser.add_argument(
        "--path_configs", default="./configs/train_downstream/default.yaml"
    )

    # Override-params for a quick resource allocation adjustment or for debugging purposes
    # If it is *not* None, the values in args override the values in the YAML file.
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--max_epochs", default=None)
    parser.add_argument("--num_workers_for_dm", default=None)
    parser.add_argument("--test_only", default=None)
    parser.add_argument("--zero_shot", default=None)
    parser.add_argument("--path_load_pretrained_imu_encoder", default=None)
    args = parser.parse_args()

    # Load the YAML file
    with open(args.path_configs) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Override the configs with args, if requested
    if args.gpus is not None:
        configs["train_hparams"]["gpus"] = int(args.gpus)
    if args.num_workers_for_dm is not None:
        configs["train_hparams"]["num_workers_for_dm"] = int(args.num_workers_for_dm)
    if args.max_epochs is not None:
        configs["train_hparams"]["max_epochs"] = int(args.max_epochs)
    if args.test_only is not None:
        configs["train_hparams"]["test_only"] = eval(args.test_only)
    if args.zero_shot is not None:
        configs["train_hparams"]["zero_shot"] = eval(args.zero_shot)
    if args.path_load_pretrained_imu_encoder is not None:
        configs["train_hparams"][
            "path_load_pretrained_imu_encoder"
        ] = args.path_load_pretrained_imu_encoder

    print(configs)
    train_downstream(configs)
