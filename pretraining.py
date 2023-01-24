# Copyright (c) Meta Platforms, Inc. and affiliates.
# LICENSE file in the root directory of this source tree.
import random
import os
from datetime import datetime
import torch
import pytorch_lightning as pl
from dataset.ego4d.dataloader import filter_narration, clean_narration_text
from lib.imu_models import MW2StackRNNPooling
from lib.clip_model import ClipPLModel
from lib.train_modules import MultimodalContrastiveLearningModule
from lib.data_modules import Ego4dDataModule, UnsupEgo4dDataModule, Split
from lib.evaluation import evaluate
from argparse import ArgumentParser
import yaml

def train(configs):

    random.seed(1234)

    # Load Model Parameters
    model_hparams = configs.get("model_hparams", {})
    model_name = model_hparams.get("model_name")
    model_suffix = model_hparams.get("model_suffix", "")
    imu_encoder_name = model_hparams.get("imu_encoder_name")
    audio_encoder_name = model_hparams.get("audio_encoder_name")
    video_encoder_name = model_hparams.get("video_encoder_name")
    window_sec = model_hparams.get("window_sec")
    target_fps = model_hparams.get("target_fps")
    datasetname = model_hparams.get("datasetname", "ego4d")
    imu_sampling_rate = model_hparams.get(
        "imu_sampling_rate", 200 if datasetname == "ego4d" else 1000
    )
    final_embedding_size = model_hparams.get("final_embedding_size", 512)

    # Params for the trainer
    train_hparams = configs.get("train_hparams", {})
    source_modality = train_hparams.get("source_modality")
    target_modalities = train_hparams.get("target_modalities")
    limit_train_batches = train_hparams.get("limit_train_batches")
    batch_size = train_hparams.get("batch_size")
    max_epochs = train_hparams.get("max_epochs")
    gpus = train_hparams.get("gpus")
    num_workers_for_dm = train_hparams.get("num_workers_for_dm")
    test_only = train_hparams.get("test_only")
    trainer_strategy = train_hparams.get("trainer_strategy")
    freeze_modalities = train_hparams.get("freeze_modalities")
    path_load_pretrained_imu_encoder = train_hparams.get(
        "path_load_pretrained_imu_encoder"
    )
    path_load_pretrained_audio_encoder = train_hparams.get(
        "path_load_pretrained_audio_encoder"
    )

    # Paths, etc.
    path_root_save_dir = f"./saved/{model_name}"
    if not os.path.exists(path_root_save_dir):
        os.makedirs(path_root_save_dir)
    target_modalities.sort()
    list_modalities = [source_modality] + target_modalities
    source_modality_initial = source_modality[0]
    target_modality_initials = "".join([m[0] for m in target_modalities])
    if source_modality == "imu":
        source_encoder_name = imu_encoder_name
    if source_modality == "audio":
        source_encoder_name = audio_encoder_name
    model_identifier = (
        f"{model_name}_s_{source_modality_initial}_t_{target_modality_initials}"
        + f"_se_{source_encoder_name}_w_{window_sec}"
    )
    if model_suffix != "":
        model_identifier += "_" + model_suffix
    else:
        model_identifier += "_%d" % (int(datetime.now().timestamp() % 10000))
    path_save_checkpoint = f"{path_root_save_dir}/{model_identifier}_best.ckpt"
    path_save_src_encoder = f"{path_root_save_dir}/{model_identifier}_src_encoder.pt"
    result_path = f"./results/{model_identifier}"
    configs["path_save_checkpoint"] = path_save_checkpoint

    # Initialize the data module
    dataset_params = {
        "window_sec": window_sec,
        "target_fps": target_fps,
        "list_modalities": list_modalities,
        "clean_narration_func": clean_narration_text,
        "filter_narration_func": filter_narration,
        "imu_sampling_rate": imu_sampling_rate,
    }

    if "text" in list_modalities:
        datamodule = Ego4dDataModule(
            batch_size=batch_size,
            num_workers=num_workers_for_dm,
            pin_memory=True,
            drop_last=True,
            dataset_params=dataset_params,
        )
    else:
        datamodule = UnsupEgo4dDataModule(
            batch_size=batch_size,
            num_workers=num_workers_for_dm,
            pin_memory=True,
            drop_last=True,
            dataset_params=dataset_params,
        )

    # Initialize encoder models
    text_encoder, video_encoder, imu_encoder = None, None, None
    modality_to_encoder = {}

    if "text" in list_modalities:
        # For now we only use a CLIP-based text model
        text_encoder = ClipPLModel(freeze=True)
        modality_to_encoder["text"] = text_encoder

    if "imu" in list_modalities:

        imu_encoder = MW2StackRNNPooling(size_embeddings=final_embedding_size)

        if path_load_pretrained_imu_encoder:
            # Load the parameters
            imu_encoder.load_state_dict(torch.load(path_load_pretrained_imu_encoder))
            print("loaded pretrained imu model")

        modality_to_encoder["imu"] = imu_encoder

    if "video" in list_modalities:
        # For now we only use a CLIP-based image model as a video encoder
        video_encoder = (
            ClipPLModel(freeze=True) if text_encoder is None else text_encoder
        )
        video_encoder.video_encoder_name = video_encoder_name

        modality_to_encoder["video"] = video_encoder

    for modality in list_modalities:
        if modality in freeze_modalities:
            modality_to_encoder[modality].eval()
            print("Freezing modality: ", modality)
            modality_to_encoder[modality].freeze()

    # Initialize the training module for contrastive training
    model = MultimodalContrastiveLearningModule(
        modality_to_encoder=modality_to_encoder,
        source_modality=source_modality,
        target_modalities=target_modalities,
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
        strategy=trainer_strategy,
        limit_train_batches=limit_train_batches,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )

    if not test_only:
        # Start training
        print("Start training: [%s] ..." % path_save_checkpoint)
        trainer.fit(model, datamodule=datamodule)

        # Save the checkpoint & encoder to a temp folder
        torch.distributed.barrier()
        print("Best checkpoint:", checkpoint_callback.best_model_path)
        model.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            modality_to_encoder=modality_to_encoder,
            source_modality=source_modality,
            target_modalities=target_modalities,
        )
        src_encoder = None
        if source_modality == "imu":
            src_encoder = model.imu_encoder
        elif source_modality == "audio":
            src_encoder = model.audio_encoder
        elif source_modality == "video":
            src_encoder = model.video_encoder
        torch.save(src_encoder.state_dict(), path_save_src_encoder)
    else:
        print("Skipping training ...")

    # Test the performance
    print("Start evaluating ...")
    metrics = evaluate(
        datamodule.get_dataset(
            "test",
            window_sample_rate=1.0,
            video_uid_sample_rate=0.25,
            max_n_windows_per_video=2,
        ),
        datamodule.collate_fn,
        model,
        source_modality,
        target_modalities,
        result_path,
        configs,
    )
    print(metrics)
    return metrics

if __name__ == "__main__":


    parser = ArgumentParser()

    # Main parameters are defined in a YAML file
    parser.add_argument(
        "--path_configs", default="./configs/train_contrastive/default.yaml"
    )

    # Override-params for a quick resource allocation adjustment or for debugging purposes
    # If it is *not* None, the values in args override the values in the YAML file.
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--num_workers_for_dm", default=None)
    parser.add_argument("--max_epochs", default=None)
    parser.add_argument("--test_only", default=None)
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
    if args.path_load_pretrained_imu_encoder is not None:
        configs["train_hparams"][
            "path_load_pretrained_imu_encoder"
        ] = args.path_load_pretrained_imu_encoder

    print(configs)
    train(configs)
