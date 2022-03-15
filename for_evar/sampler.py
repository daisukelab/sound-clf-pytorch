"""Samplers.

Mostly borrowed from:
https://github.com/qiuqiangkong/audioset_tagging_cnn
"""

import numpy as np
import logging


class BalancedRandomSampler():
    """
    This is a simple version of:
    https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4/utils/data_generator.py#L175
    """
    def __init__(self, dataset, batch_size, random_seed=42):

        self.dataset = dataset
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)

        self.samples_per_class = np.sum(self.dataset.labels.numpy(), axis=0)
        logging.info(f'samples per class: {self.samples_per_class.astype(np.int32)}')

        # Training indexes of all sound classes. E.g.: 
        # [[0, 11, 12, ...], [3, 4, 15, 16, ...], [7, 8, ...], ...]
        self.indexes_per_class = []
        self.classes_num = len(self.dataset.classes)

        for k in range(self.classes_num):
            self.indexes_per_class.append(
                np.where(dataset.labels[:, k] != 0)[0])

        # Shuffle indexes
        for k in range(self.classes_num):
            self.random_state.shuffle(self.indexes_per_class[k])

        self.queue = []
        self.pointers_of_classes = [0] * self.classes_num

    def expand_queue(self, queue):
        classes_set = np.arange(self.classes_num).tolist()
        self.random_state.shuffle(classes_set)
        queue += classes_set
        return queue

    def __iter__(self):
        while True:
            batch_idxs = []
            for _ in range(self.batch_size):
                if len(self.queue) == 0:
                    self.queue = self.expand_queue(self.queue)

                class_id = self.queue.pop(0)
                pointer = self.pointers_of_classes[class_id]
                self.pointers_of_classes[class_id] += 1
                batch_idxs.append(self.indexes_per_class[class_id][pointer])
                
                # When finish one epoch of a sound class, then shuffle its indexes and reset pointer
                if self.pointers_of_classes[class_id] >= self.samples_per_class[class_id]:
                    self.pointers_of_classes[class_id] = 0
                    self.random_state.shuffle(self.indexes_per_class[class_id])

            yield batch_idxs

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class InfiniteSampler(object):
    def __init__(self, dataset, batch_size, random_seed=42, shuffle=False):
        self.df = dataset.df
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)
        self.indexes = self.df.index.values.copy()
        self.shuffle = shuffle
        if self.shuffle:
            self.random_state.shuffle(self.indexes)

    def __iter__(self):
        pointer = 0
        while True:
            batch_idxs = []
            for _ in range(self.batch_size):
                batch_idxs.append(self.indexes[pointer])
                pointer += 1
                if pointer >= len(self.indexes):
                    pointer = 0
                    if self.shuffle:
                        self.random_state.shuffle(self.indexes)
            yield batch_idxs

    def __len__(self):
        return (len(self.df) + self.batch_size - 1) // self.batch_size
