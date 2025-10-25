import math
import random
import numpy as np
import torch
from torch.utils.data import Sampler
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler

class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, n_classes, n_samples):
        if dataset.train:
            self.labels = dataset.train_labels
        else:
            self.labels = dataset.test_labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield from indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset)

class OrderedDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples

class DistBalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, num_classes, n_sample_classes, n_samples, seed=666, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.seed = seed
        self.rank = rank
        if dataset.train:
            self.labels = dataset.train_labels
        else:
            self.labels = dataset.test_labels
        self.labels_set = list(np.arange(num_classes))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.seed(self.seed)
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_sample_classes = n_sample_classes
        self.n_samples = n_samples
        self.batch_size = self.n_samples * self.n_sample_classes
        self.total_samples_per_replica = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.total_samples_per_replica:
            classes = np.random.choice(self.labels_set, self.n_sample_classes, replace=False)
            indices = []
            for class_ in classes:
                start = self.used_label_indices_count[class_] + (self.rank % self.num_replicas)
                end = self.used_label_indices_count[class_] + self.n_samples * self.num_replicas
                step = self.num_replicas
                indices.extend(self.label_to_indices[class_][start:end:step])
                self.used_label_indices_count[class_] += self.n_samples * self.num_replicas
                if self.used_label_indices_count[class_] + self.n_samples * self.num_replicas > len(self.label_to_indices[class_]):
                    np.random.seed(self.seed)
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield from indices
            self.count += self.n_sample_classes * self.n_samples

    def __len__(self):
        return self.total_samples_per_replica