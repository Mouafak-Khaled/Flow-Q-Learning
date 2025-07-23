from fql.utils.datasets import Dataset


class InitialObservationLoader:
    def __init__(self, dataset: Dataset):
        initial_indices = range(0, len(dataset["observations"]), 1000)
        self.dataset = Dataset.create(**dataset.get_subset(initial_indices))

    def sample(self, batch_size: int):
        return self.dataset.sample(batch_size)
