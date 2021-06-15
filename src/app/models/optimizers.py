import os
from dataclasses import astuple, dataclass
from typing import Iterable

import torch
from torch.nn.parameter import Parameter
from torch.optim import Adam


@dataclass(eq=True, frozen=True)
class Optimizers:
    generator: Adam
    discriminator: Adam

    WEIGHT_FILENAME = "optimizers.pt"

    @classmethod
    def from_params(
        cls,
        generator_params: Iterable[Parameter],
        discriminator_params: Iterable[Parameter],
        learning_rate: float,
    ) -> "Optimizers":
        return cls(
            generator=Adam(generator_params, lr=learning_rate),
            discriminator=Adam(discriminator_params, lr=learning_rate),
        )

    @classmethod
    def load(cls, dir: str) -> "Optimizers":
        if not dir or not os.path.isdir(dir):
            raise FileNotFoundError("Please provide weights dir")

        path = os.path.join(dir, cls.WEIGHT_FILENAME)
        return cls(*torch.load(path))

    def save(self: "Optimizers", dir: str) -> None:
        if not dir or not os.path.isdir(dir):
            raise FileNotFoundError("Please provide weights dir")

        optimizer_path = os.path.join(dir, self.WEIGHT_FILENAME)
        torch.save(astuple(self), optimizer_path)
        return
