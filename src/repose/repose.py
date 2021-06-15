# https://github.com/tooth2/Handwritten-digits-generation/blob/main/MNIST_GAN.ipynb

import csv
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch.functional import Tensor
from torch.utils.data import DataLoader

from .models import Discriminator, Generator, Optimizers
from .utils import Loss

TRAINING_LOSS_LOG_MSG = """
Epoch [{}/{}]:
| discriminator loss: {}
|     generator loss: {}
    """


@dataclass
class Repose:
    data_length: int
    generator: Generator
    discriminator: Discriminator
    optimizers: Optimizers

    WEIGHT_FILENAME = "repose.pt"

    def __init__(
        self: "Repose",
        data_length: int = 0,
        generator: Generator = None,
        discriminator: Discriminator = None,
        optimizers: Optimizers = None,
        learning_rate: float = 0.002,
    ):
        self.data_length = data_length
        self.generator = generator if generator else Generator(data_length)
        self.discriminator = (
            discriminator if discriminator else Discriminator(data_length)
        )
        self.optimizers = (
            optimizers
            if optimizers
            else Optimizers.from_params(
                generator_params=list(self.generator.parameters()),
                discriminator_params=list(self.discriminator.parameters()),
                learning_rate=learning_rate,
            )
        )

    @classmethod
    def load(cls, dir: str) -> "Repose":
        if not dir or not os.path.isdir(dir):
            raise FileNotFoundError("Please provide weights dir")

        path = os.path.join(dir, cls.WEIGHT_FILENAME)
        return torch.load(path)

    @classmethod
    def load_weights(cls, dir: str, data_length: int) -> "Repose":
        return cls(
            data_length=data_length,
            generator=Generator.load(dir),
            discriminator=Discriminator.load(dir),
            optimizers=Optimizers.load(dir),
        )

    def generate(self: "Repose", n: int = 1) -> Tensor:
        fake_data = self.__random_tensor(n)
        return self.generator(fake_data)

    def save(self: "Repose", dir: str):
        if not dir or not os.path.isdir(dir):
            raise FileNotFoundError("Please provide weights dir")

        path = os.path.join(dir, "repose.pt")
        torch.save(self, path)
        return

    def save_weights(self: "Repose", dir: str) -> None:
        self.generator.save(dir)
        self.discriminator.save(dir)
        self.optimizers.save(dir)
        return

    def train(
        self: "Repose",
        train_loader: DataLoader,
        num_epochs: int = 1,
        save_path: Optional[str] = None,
    ):
        sample_saver = self.__get_sample_saver(save_path) if save_path else None

        for epoch in range(num_epochs):
            for batch, real_poses in enumerate(train_loader):
                batch_size = real_poses.size(0)
                discriminator_loss = self.__train_discriminator(
                    batch_size,
                    real_poses * 2 - 1,  # rescale inputs from [0, 1] to [-1, 1]
                )
                generator_loss = self.__train_generator(batch_size=batch_size)

                if sample_saver:
                    sample_saver(batch)

            print(
                TRAINING_LOSS_LOG_MSG.format(
                    epoch + 1,
                    num_epochs,
                    discriminator_loss.item(),
                    generator_loss.item(),
                )
            )

        return

    def __train_discriminator(
        self: "Repose",
        batch_size: int,
        real_poses: Tensor,
    ) -> Tensor:
        self.optimizers.discriminator.zero_grad()
        # Compute the discriminator losses on real poses
        real_loss = Loss.calc(self.discriminator(real_poses), smooth=True)
        # Compute the discriminator losses on fake poses
        fake_data = self.__random_tensor(batch_size)
        fake_poses = self.generator(fake_data)
        fake_loss = Loss.calc(self.discriminator(fake_poses), real=False)
        # add up loss and perform backprop
        d_loss = real_loss + fake_loss
        d_loss.backward()
        self.optimizers.discriminator.step()
        return d_loss

    def __train_generator(self, batch_size: int) -> Tensor:
        self.optimizers.generator.zero_grad()
        # Compute the discriminator losses on fake poses
        # use real loss to flip labels (??)
        fake_data = self.__random_tensor(batch_size)
        fake_poses = self.generator(fake_data)
        g_loss = Loss.calc(self.discriminator(fake_poses))
        g_loss.backward()  # perform backprop
        self.optimizers.generator.step()
        return g_loss

    def __random_tensor(self: "Repose", sample_size: int) -> Tensor:
        z = np.random.uniform(-1, 1, size=(sample_size, self.data_length))
        return torch.from_numpy(z).float()

    def __get_sample_saver(
        self: "Repose",
        path: str,
    ) -> Callable[[int], None]:
        sample_poses = self.__random_tensor(16)

        def _save_samples(batch: int) -> None:
            self.generator.eval()
            samples = self.generator(sample_poses)
            with open(f"{path}/sample-{batch}.csv", "w", newline="\n") as f:
                writer = csv.writer(f)
                writer.writerows(samples.detach().numpy())
            self.generator.train()
            return

        return _save_samples
