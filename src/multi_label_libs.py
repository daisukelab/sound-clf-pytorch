import pytorch_lightning as pl
import datetime
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
from dlcliche.torch_utils import IntraBatchMixupBCE
from dlcliche.utils import copy_file
from .lwlrap import Lwlrap
from skmultilearn.model_selection import IterativeStratification


class SplitAllDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, df, normalize=False):
        self.df = df
        self.normalize = normalize

        # Calculate length of clip this dataset will make
        self.L = cfg.unit_length

        # Get # of splits for all files
        self.n_splits = np.array([(np.load(f).shape[-1] + self.L - 1) // self.L for f in df.index.values])
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


def eval_all_splits(cfg, model, device, classes, df, normalize=False, debug_name=None, n=1, bs=64):
    model = model.to(device).eval()
    file_probas = [[] for _ in range(len(df))]
    test_dataset = SplitAllDataset(cfg, df, normalize=normalize)
    test_loader = torch.utils.data.DataLoader(test_dataset, num_workers=multiprocessing.cpu_count(),
                                              batch_size=bs, pin_memory=True)
    print(f'Predicting all {len(test_dataset)} splits for {len(df)} files...')
    for _ in range(n):
        with torch.no_grad():
            for X, fileidxs in test_loader:
                preds = model(X.to(device))
                probas = F.sigmoid(preds)
                for idx, proba in zip(fileidxs.cpu().numpy(), probas.cpu().numpy()):
                    file_probas[idx].append(proba)
    file_probas = np.array([np.mean(probas, axis=0) for probas in file_probas])
    lwlrap = Lwlrap(classes)
    lwlrap.accumulate(df.values, file_probas)
    return file_probas, lwlrap.overall_lwlrap(), lwlrap.per_class_lwlrap()


def sample_length(log_mel_spec):
    return log_mel_spec.shape[-1]


class MLClfDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, df, transforms=None, normalize=False):
        self.df = df
        self.transforms = transforms
        self.normalize = normalize

        # Calculate length of clip this dataset will make
        self.cfg = cfg
        self.unit_length = cfg.unit_length
        self.hop = cfg.hop_length / cfg.sample_rate

        # Show basic info.
        print(f'Dataset will yield log-mel spectrogram {len(self)} data samples in shape [1, {cfg.n_mels}, {self.unit_length}]')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        assert 0 <= index and index < len(self)
        row = self.df.iloc[index]
        filename = f'{row.name}'

        log_mel_spec = np.load(filename)

        # normalize - instance based
        if self.normalize:
            _m, _s = log_mel_spec.mean(),  log_mel_spec.std() + np.finfo(np.float).eps
            log_mel_spec = (log_mel_spec - _m) / _s

        # Padding if sample is shorter than expected - both head & tail are filled with 0s
        pad_size = self.unit_length - sample_length(log_mel_spec)
        offset = 0
        if pad_size > 0:
            offset = np.random.randint(1, pad_size) if pad_size > 1 else 0 # (pad_size // 2) -- for making it center
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 0), (offset, pad_size - offset)), 'constant')

        # Random crop
        crop_size = sample_length(log_mel_spec) - self.unit_length
        start = 0
        if crop_size > 0:
            start = np.random.randint(0, crop_size)
            log_mel_spec = log_mel_spec[..., start:start + self.unit_length]

        # Apply augmentations
        log_mel_spec = torch.Tensor(log_mel_spec)
        if self.transforms is not None:
            log_mel_spec = self.transforms(log_mel_spec)

        return log_mel_spec, row.values


class MLClfLearner(pl.LightningModule):

    def __init__(self, model, dataloaders, classes, learning_rate=3e-4, mixup_alpha=0.2, weight=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model
        self.classes = classes
        self.train_loader, self.valid_loader, self.test_loader = dataloaders

        self.criterion = nn.BCEWithLogitsLoss(weight=weight)
        self.batch_mixer = IntraBatchMixupBCE(alpha=mixup_alpha)
        self.lwlrap = Lwlrap(classes)

    def forward(self, x):
        x = self.model(x)
        return x

    def step(self, x, y, train):
        mixed_inputs, mixed_labels = self.batch_mixer.transform(x, y, train=train)
        preds = self(mixed_inputs)
        #print(preds, mixed_labels.to(torch.float))
        loss = self.criterion(preds, mixed_labels.to(torch.float))
        return preds, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds, loss = self.step(x, y, train=True)
        return loss

    def on_validation_start(self, **kwargs):
        self.lwlrap = Lwlrap(self.classes)

    def validation_step(self, batch, batch_idx, split='val'):
        x, gt = batch
        preds, loss = self.step(x, gt, train=False)
        self.lwlrap.accumulate(gt.cpu().numpy(), F.sigmoid(preds).cpu().numpy())

        self.log(f'{split}_loss', loss, prog_bar=True)
        #batch_lwlrap = lwlrap(gt.cpu().numpy(), preds.cpu().numpy())
        #self.log(f'{split}_lwlrap', batch_lwlrap, prog_bar=True)
        if batch_idx >= len(self.valid_loader) - 1:
            self.log(f'val_lwlrap', self.lwlrap.overall_lwlrap(), prog_bar=False)
        logging.info(self.lwlrap)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, split='test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader


def ml_fold_spliter(train_df, random_state=42):
    fnames = train_df.index.values

    # multi label stratified train-test splitter
    splitter = IterativeStratification(n_splits=5, random_state=random_state)

    for train, test in splitter.split(train_df.index, train_df):
        yield train_df.iloc[train], train_df.iloc[test]
