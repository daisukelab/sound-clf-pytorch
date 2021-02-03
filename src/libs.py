import warnings
warnings.simplefilter('ignore')

# Essential PyTorch
import torch
import torchaudio

# Other modules used in this notebook
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import Audio
import fire
import yaml
import multiprocessing
from easydict import EasyDict
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

import torchvision.transforms as VT
import torchaudio.transforms as AT

from dlcliche.torch_utils import IntraBatchMixup
from dlcliche.utils import copy_file

from src.augmentations import GenericRandomResizedCrop



device = torch.device('cuda')

def load_config(filename):
    with open(filename) as conf:
        cfg = EasyDict(yaml.safe_load(conf))
    cfg.unit_length = int((cfg.clip_length * cfg.sample_rate + cfg.hop_length - 1) // cfg.hop_length)
    cfg.data_root = Path(cfg.data_root)
    print(cfg)
    return cfg


def sample_length(log_mel_spec):
    return log_mel_spec.shape[-1]


class LMSClfDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, filenames, labels, transforms=None, norm_mean_std=None):
        assert len(filenames) == len(labels), f'Inconsistent length of filenames and labels.'

        self.filenames = filenames
        self.labels = labels
        self.transforms = transforms
        self.norm_mean_std = norm_mean_std

        # Calculate length of clip this dataset will make
        self.unit_length = cfg.unit_length

        # Test with first file
        assert self[0][0].shape[-1] == self.unit_length, f'Check your files, failed to load {filenames[0]}'

        # Show basic info.
        print(f'Dataset will yield log-mel spectrogram {len(self)} data samples in shape [1, {cfg.n_mels}, {self.unit_length}]')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        assert 0 <= index and index < len(self)
        
        log_mel_spec = np.load(self.filenames[index])
        
        # normalize - instance based
        if self.norm_mean_std is not None:
            log_mel_spec = (log_mel_spec - self.norm_mean_std[0]) / self.norm_mean_std[1]

        # Padding if sample is shorter than expected - both head & tail are filled with 0s
        pad_size = self.unit_length - sample_length(log_mel_spec)
        if pad_size > 0:
            offset = pad_size // 2
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 0), (offset, pad_size - offset)), 'constant')

        # Random crop
        crop_size = sample_length(log_mel_spec) - self.unit_length
        if crop_size > 0:
            start = np.random.randint(0, crop_size)
            log_mel_spec = log_mel_spec[..., start:start + self.unit_length]

        # Apply augmentations
        log_mel_spec = torch.Tensor(log_mel_spec)
        if self.transforms is not None:
            log_mel_spec = self.transforms(log_mel_spec)

        return log_mel_spec, self.labels[index]


class SplitAllDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, df, normalize=False, top_n=99999):
        self.df = df
        self.normalize = normalize

        # Calculate length of clip this dataset will make
        self.L = cfg.unit_length

        # Get # of splits for all files
        self.n_splits = np.array([(np.load(f).shape[-1] + self.L - 1) // self.L for f in df.index.values])
        self.n_splits = np.clip(1, top_n, self.n_splits) # limit number of splits.
        self.sum_splits = np.cumsum(self.n_splits)

    def __len__(self):
        return self.sum_splits[-1]

    def file_index(self, index):
        return sum((index < self.sum_splits) == False)

    def filename(self, index):
        return self.df.index.values[self.file_index(index)]

    def split_index(self, index):
        fidx = self.file_index(index)
        prev_sum = self.sum_splits[fidx - 1] if fidx > 0 else 0
        return index - prev_sum

    def __getitem__(self, index):
        assert 0 <= index and index < len(self)

        log_mel_spec = np.load(self.filename(index))
        start = self.split_index(index) * self.L
        log_mel_spec = log_mel_spec[..., start:start + self.L]

        # normalize - instance based
        if self.normalize:
            _m, _s = log_mel_spec.mean(),  log_mel_spec.std() + np.finfo(np.float).eps
            log_mel_spec = (log_mel_spec - _m) / _s

        # Padding if sample is shorter than expected - both head & tail are filled with 0s
        pad_size = self.L - sample_length(log_mel_spec)
        if pad_size > 0:
            offset = pad_size // 2
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 0), (offset, pad_size - offset)), 'constant')

        return log_mel_spec, self.file_index(index)


class LMSClfLearner(pl.LightningModule):

    def __init__(self, model, dataloaders, learning_rate=3e-4, mixup_alpha=0.0, weight=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model
        self.trn_dl, self.val_dl, self.test_dl = dataloaders
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.batch_mixer = IntraBatchMixup(self.criterion, alpha=mixup_alpha) if mixup_alpha > 0.0 else None

    def forward(self, x):
        x = self.model(x)
        return x

    def step(self, x, y, train):
        if self.batch_mixer is None:
            preds = self(x)
            loss = self.criterion(preds, y)
        else:
            x, stacked_y = self.batch_mixer.transform(x, y, train=train)
            preds = self(x)
            loss = self.batch_mixer.criterion(preds, stacked_y)
        return preds, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds, loss = self.step(x, y, train=True)
        return loss

    def validation_step(self, batch, batch_idx, split='val'):
        x, y = batch
        preds, loss = self.step(x, y, train=False)
        yhat = torch.argmax(preds, dim=1)
        acc = accuracy(yhat, y)

        self.log(f'{split}_loss', loss, prog_bar=True)
        self.log(f'{split}_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, split='test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return self.trn_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl