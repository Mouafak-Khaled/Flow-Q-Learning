from hpo.strategy import HpoStrategy


class IdentityStrategy(HpoStrategy):
    """
    A strategy that returns the entire population without any modifications.
    """

    def update(self, candidate, performance: float) -> None:
        pass

    def sample(self) -> list:
        return self.population
