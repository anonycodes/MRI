import math
import os
from collections import namedtuple
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from joblib import dump, load
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, sampler
from torch.utils.data.dataset import ConcatDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from .utils import MultiDataset


class KneeDataset(MultiDataset):
    def __init__(
        self,
        split_csv_file: str,
        mode: str,
        label_type: str,
        data_space: str,
        noise_percent: float,
        coil_type="sc",
        image_only: bool = False,
    ):
        super().__init__(split_csv_file=split_csv_file, mode=mode)
        fields = [
            "volume_id",
            "slice_id",
            "sc_kspace",
            "mc_kspace",
            "recon_esc",
            "recon_rss",
            "label",
            "data_split",
            "dataset",
            "location",
            "max_value",
        ]
        self.sample_template = namedtuple(
            "Sample", fields, defaults=(math.nan,) * len(fields)
        )
        self.coil_type = coil_type
        assert self.coil_type in {"sc", "mc"}

        self.image_only = image_only
        self.label_type = label_type
        self.data_space = data_space
        self.noise_percent = noise_percent
        self.mode = mode

    def parse_label(self, label_arr: Sequence[str]) -> torch.Tensor:
        label_arr = label_arr.replace("[", "").replace("]", "").replace("'", "")
        label_arr = label_arr.split(",")

        if "None" in label_arr:
            return torch.Tensor([0.0, 0.0, 0.0, 0.0]).float()

        new_labels = []

        for label in label_arr:
            if "ACL" in label:
                new_labels.append(1)
            elif "Meniscus Tear" in label:
                new_labels.append(2)
            elif "cartilage" in label.lower():
                new_labels.append(3)
            else:
                new_labels.append(4)

        abnormal = 1.0 if 4 in new_labels else 0.0
        cartilage = 1.0 if 3 in new_labels else 0.0
        mtear = 1.0 if 2 in new_labels else 0.0
        acl = 1.0 if 1 in new_labels else 0.0

        return torch.Tensor(np.array([abnormal, mtear, acl, cartilage])).float()

    def parse_multilabel(self, label_arr: Sequence[str]) -> torch.Tensor:
        label_arr = label_arr.replace("[", "").replace("]", "").replace("'", "")
        label_arr = label_arr.split(",")

        if "None" in label_arr:
            return dict(
                acl=torch.zeros(1).long(),
                mtear=torch.zeros(1).long(),
                edema=torch.zeros(1).long(),
                fracture=torch.zeros(1).long(),
                cartilage=torch.zeros(1).long(),
                effusion=torch.zeros(1).long(),
                cysts=torch.zeros(1).long(),
            )

        new_labels = []

        for label in label_arr:
            if "ACL" in label:
                new_labels.append(1)
            elif "Meniscus Tear" in label:
                new_labels.append(2)
            elif "edema" in label.lower():
                new_labels.append(3)
            elif "fracture" in label.lower():
                new_labels.append(4)
            elif "cartilage" in label.lower():
                new_labels.append(5)
            elif "joint effusion " in label.lower():
                new_labels.append(6)
            elif "cysts" in label.lower():
                new_labels.append(7)

        acl = 1.0 if 1 in new_labels else 0.0
        mtear = 1.0 if 2 in new_labels else 0.0
        edema = 1.0 if 3 in new_labels else 0.0
        fracture = 1.0 if 4 in new_labels else 0.0
        cartilage = 1.0 if 5 in new_labels else 0.0
        effusion = 1.0 if 6 in new_labels else 0.0
        cysts = 1.0 if 7 in new_labels else 0.0

        labels = dict(
            acl=torch.Tensor([acl]).long(),
            mtear=torch.Tensor([mtear]).long(),
            edema=torch.Tensor([edema]).long(),
            fracture=torch.Tensor([fracture]).long(),
            cartilage=torch.Tensor([cartilage]).long(),
            effusion=torch.Tensor([effusion]).long(),
            cysts=torch.Tensor([cysts]).long(),
        )
        return labels

    def __getitem__(self, index):
        assert self.mode in self.metadata_by_mode
        loc = self.get_metadata_value(index, "location")

        info = self.metadata_by_mode[self.mode].iloc[index]
        kspace_key = "sc_kspace" if self.coil_type == "sc" else "mc_kspace"
        target_key = "recon_esc" if self.coil_type == "sc" else "recon_rss"

        with h5py.File(loc) as f:
            kspace_data = f[kspace_key][:]
            target_data = f[target_key][:]

            image_data = torch.view_as_real(torch.from_numpy(kspace_data))
            from fastmri.fftc import ifft2c_new, fft2c_new
            image_data = ifft2c_new(image_data)
            
            from fastmri.math import complex_abs
            signal_strength = torch.max(complex_abs(image_data))
            noise_perc = torch.randint(low=0, high=self.noise_percent, size=(1,)) if self.noise_percent != 0. else 0.
            noise_percent = noise_perc if self.mode == "train" else self.noise_percent

            std = signal_strength * (noise_percent/100)
            if noise_percent > 0.:
                image_data[:,:,0] += torch.normal(mean=0., std=std, size=image_data[:,:,0].shape)
                image_data[:,:,1] += torch.normal(mean=0., std=std, size=image_data[:,:,0].shape)
                kspace_data = torch.view_as_complex(image_data)
            else:
                kspace_data = torch.view_as_complex(image_data)

            parameters = {
                kspace_key: kspace_data,
                target_key: target_data,
                "volume_id": info.volume_id,
                "slice_id": info.slice_id,
                "data_split": info.data_split,
                "dataset": info.dataset,
                "location": info.location,
            }
            if self.label_type == "knee_multilabel":
                parameters["label"] = self.parse_multilabel(info.labels)
            elif self.label_type == "knee":
                parameters["label"] = self.parse_label(info.labels)
            else:
                raise NotImplementedError(
                    f"Label type {self.label_type} not implemented"
                )                

        sample = self.sample_template(**parameters)
        return sample


def get_sampler_weights(dataset, save_filename="./sampler_knee_tr.p"):
    Y_tr = []

    for i in tqdm(range(len(dataset))):
        label = dataset[i].label.sum().item()
        Y_tr.append(label)

    Y_tr = np.array(Y_tr).astype(int)

    class_sample_count = np.array(
        [len(np.where(Y_tr == t)[0]) for t in np.unique(Y_tr)]
    )

    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in Y_tr])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    dump(sampler, save_filename)


class KneeDataClassificationModule(pl.LightningDataModule):
    def __init__(
        self,
        split_csv_file: str,
        label_type: str,
        coil_type: str,
        batch_size: int,
        noise_percent: int,
        data_space: str,
        sampler_filename: Optional[str] = None,
        combine_class_recon: bool = False,
        dev_mode: bool = False,
        num_workers: int = 0,
    ):
        super().__init__()

        self.split_csv_file = split_csv_file
        self.coil_type = coil_type
        self.batch_size = batch_size
        self.sampler_filename = sampler_filename
        self.dev_mode = dev_mode
        self.num_workers = num_workers
        self.data_space = data_space
        self.label_type = label_type
        self.noise_percent = noise_percent

    def setup(self, stage: Optional[str] = None):
        # get data split names
        test_mode: Optional[str]
        train_mode = "train_class"
        val_mode = "val_class"
        test_mode = "test_class"

        # initialize datasets
        self.train_dataset = KneeDataset(
            split_csv_file=self.split_csv_file,
            coil_type=self.coil_type,
            mode=train_mode,
            label_type=self.label_type,
            data_space=self.data_space,
            noise_percent=self.noise_percent,
        )
        self.val_dataset = KneeDataset(
            split_csv_file=self.split_csv_file,
            coil_type=self.coil_type,
            mode=val_mode,
            label_type=self.label_type,
            data_space=self.data_space,
            noise_percent=self.noise_percent,
        )

        self.test_dataset = KneeDataset(
                split_csv_file=self.split_csv_file,
                mode=test_mode,
                coil_type=self.coil_type,
                label_type=self.label_type,
                data_space=self.data_space,
                noise_percent=self.noise_percent,
        )
        # create sampler if task is classification
        if self.sampler_filename is None:
            print("Creating sampler weights...")
            self.sampler_filename = "../knee/sampler_knee_tr.p"
            get_sampler_weights(self.train_dataset, self.sampler_filename)
        elif not os.path.exists(self.sampler_filename):
            raise ValueError("Weighted sampler does not exist")
        assert Path(self.sampler_filename).is_file()
            # load the sampler
        self.train_sampler = load(self.sampler_filename)

    def train_dataloader(self) -> DataLoader:
            return DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    sampler=self.train_sampler,
            )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

