# https://github.com/tooth2/Handwritten-digits-generation/blob/main/MNIST_GAN.ipynb

import pickle
from dataclasses import dataclass

import numpy as np
import torch
from torch.functional import Tensor
from torch.utils.data import DataLoader

from src.models import Discriminator, Generator
from src.utils import Loss, Optimizers

POSE_DATA_LENGTH = 17 * 2  # aka 17 coordinates


@dataclass
class Repose:
    generator: Generator
    discriminator: Discriminator
    optimizers: Optimizers
    samples: list[Tensor]

    def __init__(
        self: "Repose",
        generator: Generator = None,
        discriminator: Discriminator = None,
        optimizers: Optimizers = None,
        learning_rate: float = 0.002,
    ):
        self.generator = generator if generator else Generator(POSE_DATA_LENGTH)
        self.discriminator = (
            discriminator if discriminator else Discriminator(POSE_DATA_LENGTH)
        )
        self.optimizers = (
            optimizers
            if optimizers
            else Optimizers.from_params(
                generator_params=self.generator.parameters(),
                discriminator_params=self.discriminator.parameters(),
                learning_rate=learning_rate,
            )
        )
        self.samples = []

    @classmethod
    def load_weights(cls, dir: str) -> "Repose":
        return cls(
            generator=Generator.load(dir),
            discriminator=Discriminator.load(dir),
            optimizers=Optimizers.load(dir),
        )

    def train(
        self: "Repose",
        train_loader: DataLoader,
        num_epochs: int = 1,
        print_frequency: int = None,
    ):
        # Get some fixed data for sampling. These are poses that are held
        # constant throughout training and allow us to inspect the model's
        # performance
        sample_poses = Repose.random_tensor(16)

        for epoch in range(num_epochs):
            for batch_i, (real_poses, _) in enumerate(train_loader):
                batch_size = real_poses.size(0)
                discriminator_loss = self.__train_discriminator(
                    batch_size,
                    Repose.rescale(real_poses),
                )
                generator_loss = self.__train_generator(batch_size=batch_size)

                if print_frequency and batch_i % print_frequency == 0:
                    print(
                        f"""
                        Epoch [{epoch + 1}/{num_epochs}]:
                            | discriminator loss: {discriminator_loss.item()}
                            | generator loss:     {generator_loss.item()}
                        -----
                        """
                    )

            # generate and save sample, fake poses
            self.generator.eval()  # eval mode for generating samples
            self.samples.append(self.generator(sample_poses))
            self.generator.train()  # back to train mode
        return

    def save_weights(self: "Repose", dir: str) -> None:
        self.generator.save(dir)
        self.discriminator.save(dir)
        self.optimizers.save(dir)
        return

    def save_samples(self: "Repose", path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.samples, f)
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
        fake_poses = self.generator(Repose.random_tensor(batch_size))
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
        fake_poses = self.generator(Repose.random_tensor(batch_size))
        g_loss = Loss.calc(self.discriminator(fake_poses))
        g_loss.backward()  # perform backprop
        self.optimizers.generator.step()
        return g_loss

    @staticmethod
    def rescale(inp):
        # rescale inputs from [0, 1] to [-1, 1]
        return inp * 2 - 1

    @staticmethod
    def random_tensor(sample_size: int) -> Tensor:
        z_size = 100
        z = np.random.uniform(-1, 1, size=(sample_size, z_size))
        return torch.from_numpy(z).float()
