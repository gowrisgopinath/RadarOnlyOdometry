# Copyright (C) 2024 Jan Michalczyk, Control of Networked Systems, University
# of Klagenfurt, Austria.
#
# All rights reserved.
#
# This software is licensed under the terms of the BSD-2-Clause-License with
# no commercial use allowed, the full terms of which are made available
# in the LICENSE file. No license in patents is granted.
#
# You can contact the author at <jan.michalczyk@aau.at>

import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path

# implements a torch dataset which loads the created
# in advance hdf5 files with labels and examples.


class HDF5Dataset(Dataset):
    def __init__(self, examples_path, labels_path):
        super(HDF5Dataset, self).__init__()
        self.examples_path = examples_path
        self.labels_path = labels_path

        if Path(self.examples_path).is_file() and Path(self.labels_path).is_file():
            self.pointclouds_file = h5py.File(self.examples_path, 'a')
            self.labels_file = h5py.File(self.labels_path, 'a')
        else:
            raise IOError("Examples and labels do not exist.")

        self.examples_dset = self.pointclouds_file["examples"]
        self.labels_dset = self.labels_file["labels"]

        if not (self.labels_dset.shape[0] == self.examples_dset.shape[0]):
            raise ValueError("Examples and labels have different lengths.")

        self.normalizer = None

    def get_input_length(self):
        return self.examples_dset.shape[1]

    def __getitem__(self, index):
        example = torch.from_numpy(self.examples_dset[index])
        label = torch.from_numpy(self.labels_dset[index])
        if self.normalizer:
            example, label = self.normalizer.normalize_data(example, label)
        example = example.unsqueeze(0)
        label = label.unsqueeze(0)
        return (example, label)

    def __len__(self):
        return self.labels_dset.shape[0]