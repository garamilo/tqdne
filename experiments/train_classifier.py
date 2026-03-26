import logging

import torch
from config import SpectrogramClassificationConfig
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from tqdne.classifier import LithningClassifier
from tqdne.dataloader import get_train_and_val_loader
from tqdne.training import get_pl_trainer
from tqdne.utils import get_device, get_last_checkpoint

def run(args):
    name = "Classifier-LogSpectrogram"
    config = SpectrogramClassificationConfig(args.workdir)
    config.representation.disable_multiprocessing()

    train_loader, val_loader = get_train_and_val_loader(
        config,
        args.num_workers,
        args.batchsize,
        cond=True,
        train_split="train_validation",
        val_split="test"
    )

    # loss and metrics
    num_classes = (len(config.mag_bins) - 1) * (len(config.dist_bins) - 1)
    loss = torch.nn.CrossEntropyLoss()
    metrics = [
        MulticlassAccuracy(num_classes),
        MulticlassRecall(num_classes),
        MulticlassPrecision(num_classes),
        MulticlassF1Score(num_classes),
    ]

    # Parameters
    encoder_config = {
        "in_channels": config.channels,
        "model_channels": 64,
        "channel_mult": (1, 2, 4, 4),
        "out_channels": 256,
        "num_res_blocks": 2,
        "attention_resolutions": (8,),
        "dims": 2,
        "conv_kernel_size": 3,
        "num_heads": 4,
        "dropout": 0.3,
        "flash_attention": False,
    }

    optimizer_params = {
        "learning_rate": 0.000025,
        "max_steps": 110 * len(train_loader),
        "eta_min": 0.0,
    }
    trainer_params = {
        "precision": 32,
        "accelerator": get_device(),
        "devices": args.num_devices,
        "num_nodes": 1,
        "num_sanity_val_steps": 0,
        "max_steps": optimizer_params["max_steps"],
    }

    logging.info("Build lightning module...")
    classifier = LithningClassifier(
        encoder_config=encoder_config,
        num_classes=num_classes,
        loss=loss,
        metrics=metrics,
        optimizer_params=optimizer_params,
    )

    logging.info("Build Pytorch Lightning Trainer...")
    trainer = get_pl_trainer(
        name=name,
        val_loader=val_loader,
        config=config,
        metrics=[],
        plots=[],
        eval_every=5,
        limit_eval_batches=10,
        log_to_wandb=True,
        **trainer_params,
    )

    logging.info("Start training...")
    torch.set_float32_matmul_precision("high")
    checkpoint = get_last_checkpoint(trainer.default_root_dir)
    trainer.fit(
        classifier,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=checkpoint,
    )

    logging.info("Done!")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser("Train a classifier")
    parser.add_argument(
        "--workdir",
        type=str,
        help="the working directory in which checkpoints and all output are saved to",
    )
    parser.add_argument(
        "-b", "--batchsize", type=int, help="size of a batch of each gradient step", default=128
    )
    parser.add_argument(
        "-w", "--num-workers", type=int, help="number of separate processes for file/io", default=32
    )
    parser.add_argument(
        "-d", "--num-devices", type=int, help="number of CPUs/GPUs to train on", default=4
    )
    args = parser.parse_args()
    if args.workdir is None:
        parser.print_help()
        sys.exit(0)
    run(args)
