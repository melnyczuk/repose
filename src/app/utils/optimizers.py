import os
from dataclasses import dataclass
from typing import Iterable

import torch
from torch.nn.parameter import Parameter
from torch.optim import Adam


@dataclass(eq=True, frozen=True)
class Optimizers:
    generator: Adam
    discriminator: Adam

    GENERATOR_OPTIMIZER_WEIGHT_FILENAME = "generator_opti.pt"
    DISCRIMINATOR_OPTIMIZER_WIGHT_FILENAME = "discriminator_opti.pt"

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

        return cls(
            generator=torch.load(
                os.path.join(
                    dir,
                    cls.GENERATOR_OPTIMIZER_WEIGHT_FILENAME,
                )
            ),
            discriminator=torch.load(
                os.path.join(
                    dir,
                    cls.DISCRIMINATOR_OPTIMIZER_WIGHT_FILENAME,
                )
            ),
        )

    def save(self: "Optimizers", dir: str) -> None:
        if not dir or not os.path.isdir(dir):
            raise FileNotFoundError("Please provide weights dir")

        generator_optimizer_path = os.path.join(
            dir,
            self.GENERATOR_OPTIMIZER_WEIGHT_FILENAME,
        )

        discriminator_optimizer_path = os.path.join(
            dir,
            self.DISCRIMINATOR_OPTIMIZER_WIGHT_FILENAME,
        )

        torch.save(self.generator, generator_optimizer_path)
        torch.save(self.discriminator, discriminator_optimizer_path)
        return
