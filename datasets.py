import numpy as np
import h5py
import torch
import glob
import os

from torch.utils.data import Dataset


class HDF5Dataset(Dataset):

    def __init__(self, folder, labels, feature, random_shift=False, sample_length=150,
                                                shift=15,
                                                transform=None,
                                                in_memory=False):

        self._files = glob.glob(os.path.join(folder, '*.hdf5'))
        self._in_memory = in_memory
        self._sample_length = sample_length
        self._shift = shift
        self._feature = feature
        self._transform = transform
        self._random_shift = random_shift

        self._data = []
        self.frms_per_seq = []
        targets = []
        for index, filename in enumerate(self._files):

            label = filename.split('/')[-1].split('_')[0]
            if label not in labels:
                continue

            with h5py.File(filename, 'r') as file:

                self.frms_per_seq.append(file[self._feature].shape[0])

                targets.append(labels.index(label))
                if self._in_memory:
                    self._data.append(file[self._feature][:])



        self._indices = []
        self._targets = []
        self._saved_data = {}
        for fi in range(len(self.frms_per_seq)):
            for si in range(0, max(1, self.frms_per_seq[fi] - 45), self._shift):
            #for si in range(0, max(1, self.frms_per_seq[fi] - self._sample_length), self._shift):
                self._indices.append((fi, si))
                self._targets.append(targets[fi])

        self._targets = torch.from_numpy(np.asarray(self._targets))
        self._length = len(self._indices)


    def __getitem__(self, index):

        file_index, sequence_index = self._indices[index]

        if self._random_shift:
            sequence_index = np.random.randint(0, self.frms_per_seq[file_index] - self._sample_length)

        if self._in_memory:
            sample = self._data[file_index][sequence_index: sequence_index + self._sample_length]
        else:
            with h5py.File(self._files[file_index], 'r') as file:
                sample = file[self._feature][sequence_index:sequence_index + self._sample_length]

        if len(sample) < self._sample_length:
            sample = np.concatenate((sample, np.repeat(sample[-1][np.newaxis], self._sample_length - len(sample), axis=0)))

        if self._transform is not None:
            sample = self._transform(sample)

        return sample, self._targets[index]

    def __len__(self):
        return self._length
