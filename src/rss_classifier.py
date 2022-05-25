import argparse
import os
import pathlib
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from tqdm.auto import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
import sys
import inspect

from data.knee_data import KneeDataClassificationModule
from rss_module import RSS
import fire
import itertools


def get_data(args: argparse.ArgumentParser) -> pl.LightningDataModule:
    # get datamodule
    if args.data_type == "knee":
        # load mc data to obtain rss images
        if args.task == "classification":
            datamodule = KneeDataClassificationModule(
                label_type="knee",
                split_csv_file=args.split_csv_file,
                coil_type=args.coil_type,
                batch_size=args.batch_size,
                sampler_filename=args.sampler_filename,
                noise_percent=args.noise_percent,
                data_space=args.data_space,
        )
    else:
        raise NotImplementedError

    return datamodule


def get_model(
    args: argparse.ArgumentParser, device: torch.device,
) -> pl.LightningModule:
    if args.data_type == "knee":
        # get spatial domain model
        model = RSS(
            model_type=args.model_type,
            data_type=args.data_type,
            image_shape=[320, 320],
            drop_prob=args.drop_prob,
            kspace_shape=[640, 400],
            data_space=args.data_space,
            device=device,
            lr=args.lr,
            weight_decay=args.weight_decay,
            lr_gamma=args.lr_gamma,
            lr_step_size=args.lr_step_size,
        )
    else:
        raise NotImplementedError
    return model

def train_model(
    args: argparse.Namespace,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    device: torch.device,
) -> pl.LightningModule:
    log_dir = (
        Path(args.log_dir)
        / args.data_type
        / args.data_space
        / f'train_noise-{args.noise_percent}'
    )
    model_dir = str(args.model_dir) + '/' + args.data_space + '/' + str(args.n_seed) + '/' + 'noise-' + str(args.noise_percent) 

    if not os.path.isdir(str(log_dir)):
        os.makedirs(str(log_dir))
    if not os.path.isdir(str(model_dir)):
        os.makedirs(str(model_dir))

    csv_logger = CSVLogger(save_dir=log_dir, name=f"train_noise-{args.noise_percent}", version=f"{args.n_seed}")
    wandb_logger = WandbLogger(name=f"{args.data_space}-{args.n_seed}")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model_checkpoint = ModelCheckpoint(monitor='val_auc_mean', dirpath=model_dir, filename="{epoch:02d}-{val_auc_mean:.2f}",save_top_k=1, mode='max')
    early_stop_callback = EarlyStopping(monitor='val_auc_mean', patience=5, mode='max')

    trainer: pl.Trainer = pl.Trainer(
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=args.n_epochs,
        logger=[wandb_logger, csv_logger],
        callbacks=[model_checkpoint, early_stop_callback, lr_monitor],
        auto_lr_find=True,
    )

    trainer.tune(model, datamodule)
    trainer: pl.Trainer = pl.Trainer(
        gpus=1 if str(device).startswith("cuda") else 0,
        max_epochs=args.n_epochs,
        logger=[wandb_logger, csv_logger],
        callbacks=[model_checkpoint, early_stop_callback, lr_monitor],
    )
    trainer.fit(model, datamodule)

    return model


def test_model(
    args: argparse.Namespace,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    device: torch.device,
) -> pl.LightningModule:
    
    noise_arr = [0]
    
    for noise in noise_arr:
        model_dir = str(args.model_dir) + '/' + args.data_space + '/'  + str(args.n_seed) + '/' + 'noise-' + str(noise)
        checkpoint_filename = os.listdir(model_dir)[0]
        print("Checkpoint file: ", model_dir, checkpoint_filename)
        log_dir = (
        Path(args.log_dir)
        / args.data_type
        / args.data_space
        / f'train_noise-{noise}'
        )

        csv_logger = CSVLogger(save_dir=log_dir, name=f"test_noise-{args.noise_percent}", version=f"{args.n_seed}")

        model = RSS.load_from_checkpoint(model_dir + '/' + checkpoint_filename)
        trainer = pl.Trainer(gpus=1 if str(device).startswith("cuda") else 0, logger=csv_logger)
        with torch.inference_mode():
            model.eval()
            M_val = trainer.validate(model, datamodule.val_dataloader())  
            M = trainer.test(model, datamodule.test_dataloader())


def get_args():
    parser = argparse.ArgumentParser(description="Indirect MR Screener training")
    # logging parameters
    parser.add_argument("--model_dir", type=str, default="../trained_models")
    parser.add_argument("--log_dir", type=str, default="../trained_logs")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--dev_mode", action="store_true")
    
    # data parameters
    parser.add_argument(
        "--data_type", type=str, default="knee",
    )
    parser.add_argument(
        "--data_space", type=str, default="ktoi_w_phase")
    parser.add_argument(
        "--task", type=str, default="classification",
    )
    parser.add_argument("--image_shape", type=int, default=[320, 320], nargs=2, required=False)
    parser.add_argument("--image_type", type=str, default='orig', required=False, choices=["orig"])
    parser.add_argument("--split_csv_file", type=str, default='..//metadata_knee.csv', required=False)
    
    parser.add_argument("--recon_model_ckpt", type=str)
    parser.add_argument("--recon_model_type", type=str, default=["rss"], required=False, choices=["rss"])
    parser.add_argument("--mask_type", type=str, default="none")
    parser.add_argument("--k_fraction", type=float, default=0.25)
    parser.add_argument("--center_fraction", type=float, default=0.08)
    parser.add_argument("--coil_type", type=str, default="sc", choices=["sc","mc"])

    parser.add_argument("--sampler_filename", type=str, default=None)
    parser.add_argument(
        "--model_type",
        type=str,
        default="preact_resnet18",
        choices=[
            "preact_resnet18",
            "preact_resnet34",
            "preact_resnet50",
            "preact_resnet101",
            "preact_resnet152",
            "pretrained_imagenet"
        ],
    )

    # training parameters
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--n_seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--drop_prob", type=float, default=0.5)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_step_size", type=int, default=5)
    parser.add_argument("--noise_percent", type=int, default=0)

    parser.add_argument("--n_masks", type=int, default=100)

    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--sweep_step", type=int)
    parser.add_argument('--debug',  default=True)

    args, unkown = parser.parse_known_args()
    
    return args


def retreve_config(args, sweep_step=None):
    grid = {
        "noise_percent": [0],
        # "noise_percent": [2, 4, 5, 10, 15, 20, 25],
        "n_seed": [1,2,3],
    }

    grid_setups = list(
        dict(zip(grid.keys(), values)) for values in itertools.product(*grid.values())
    )
    step_grid = grid_setups[sweep_step - 1]  # slurm var will start from 1

    # automatically choose the device based on the given node
    if torch.cuda.device_count() > 0:
        expr_device = "cuda"
    else:
        expr_device = "cpu"
    
    args.sweep_step = sweep_step
    args.noise_percent = step_grid["noise_percent"]
    args.n_seed = step_grid['n_seed']

    return args


def run_experiment(args):
    
    print(args, flush=True)
    if torch.cuda.is_available():
        print("Found CUDA device, running job on GPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datamodule = get_data(args)
    model = get_model(args, device)
    if args.mode == "train":
        model = train_model(args=args, model=model, datamodule=datamodule, device=device,)
    else:
        datamodule.setup()
        test_model(args=args, model=model, datamodule=datamodule, device=device,)

def main(sweep_step=None):
    args = get_args()
    run_experiment(args)


if __name__ == "__main__":
     fire.Fire(main)
