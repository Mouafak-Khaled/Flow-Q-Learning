from abc import ABC, abstractmethod

import numpy as np

from fql.utils.datasets import Dataset


class DataLoader(ABC):
    @abstractmethod
    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        pass


class InitialObservationLoader(DataLoader):
    def __init__(self, dataset: dict[str, np.ndarray]):
        initial_indices = range(0, len(dataset["observations"]), 1000)
        for key in dataset:
            dataset[key] = dataset[key][initial_indices]
        self.dataset = Dataset.create(**dataset)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        return self.dataset.sample(batch_size)


class EpisodeLoader(DataLoader):
    def __init__(self, dataset: dict[str, np.ndarray]):
        n = len(dataset["observations"])
        for key in dataset:
            dataset[key] = np.squeeze(dataset[key].reshape(n // 1000, 1000, -1))
        self.dataset = Dataset.create(**dataset)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        return self.dataset.sample(batch_size)


class StepLoader(DataLoader):
    def __init__(self, dataset: dict[str, np.ndarray]):
        self.dataset = Dataset.create(**dataset)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        return self.dataset.sample(batch_size)
