from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import torch.utils.data as torch_data
import torchvision.datasets.utils as tv_datasets_utils

DATA_ROOT = Path('data/toy/')


class _BaseToyDataset(torch_data.Dataset):
    @property
    def dataset_path(self):
        return DATA_ROOT / self.FILENAME

    def __getitem__(self, index):
        x, y = self._get_item(index)

        if x.shape[-1] == 3:
            x = np.swapaxes(x, -1, 0)
        x = x / 255. - .5

        return x, y

    @abstractmethod
    def _getitem(self, index):
        raise NotImplementedError()


class DSpritesDataset(_BaseToyDataset):
    URL = f'https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true'
    FILENAME = 'dsprites.npz'

    def __init__(self):
        tv_datasets_utils.download_url(
            url=self.URL,
            filename=self.FILENAME,
            root=DATA_ROOT,
        )

        dataset_zip = np.load(self.dataset_path)
        self.imgs = dataset_zip['imgs']
        # skip the first constant latent (color)
        self.latents_values = dataset_zip['latents_values'][:, 1:]
        self.latents_classes = dataset_zip['latents_classes'][:, 1:]

    def __len__(self):
        return self.imgs.shape[0]

    def _getitem(self, index):
        return self.imgs[index], self.latents_values[index]


class Shapes3dDataset(_BaseToyDataset):
    URL = 'https://storage.cloud.google.com/3d-shapes/3dshapes.h5'
    FILENAME = '3dshapes.h5'

    def __init__(self, in_memory=False):
        assert self.dataset_path.exists(), f'Download the dataset manually from:\n{self.URL}\nto:\n{self.dataset_path}'

        self.in_memory = in_memory
        with self._open_h5_file() as h5_file:
            self.labels = h5_file['labels'][:]
            # standardize the labels to a z-score
            self.labels = (self.labels - self.labels.mean(axis=0)) / self.labels.std(axis=0)
            if self.in_memory:
                self.images = h5_file['images'][:]

    @contextmanager
    def _open_h5_file(self):
        h5_file = h5py.File(self.dataset_path, mode='r')
        yield h5_file
        h5_file.close()

    def _get_item(self, index):
        if self.in_memory:
            return self.images[index], self.labels[index]
        else:
            with self._open_h5_file() as h5_file:
                return h5_file['images'][index], self.labels[index]

    def __len__(self):
        return self.labels.shape[0]
