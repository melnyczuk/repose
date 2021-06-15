# https://github.com/nbertagnolli/pytorch-simple-gan/blob/master/models.py
import os
from dataclasses import dataclass

import torch
from torch.functional import Tensor
from torch.nn import Linear, Module, Sigmoid


@dataclass
class Discriminator(Module):
    dense_layer: Linear
    activation: Sigmoid

    WEIGHT_FILENAME = "discriminator.pt"

    def __init__(self: "Discriminator", length: int):
        super(Discriminator, self).__init__()
        self.dense_layer = Linear(length, 1)
        self.activation = Sigmoid()

    def __hash__(self) -> int:
        return super().__hash__()

    def forward(self: "Discriminator", x: Tensor) -> Tensor:
        return self.activation(self.dense_layer(x))

    @classmethod
    def load(cls, dir: str) -> "Discriminator":
        if not dir or not os.path.isdir(dir):
            raise FileNotFoundError("Please provide weights dir")

        path = os.path.join(dir, cls.WEIGHT_FILENAME)
        return torch.load(path)

    def save(self: "Discriminator", dir: str) -> None:
        if not dir or not os.path.isdir(dir):
            raise FileNotFoundError("Please provide weights dir")

        path = os.path.join(dir, self.WEIGHT_FILENAME)
        torch.save(self, path)
        return
