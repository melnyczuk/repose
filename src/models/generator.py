# https://github.com/nbertagnolli/pytorch-simple-gan/blob/master/models.py

import os

import torch
from torch.functional import Tensor
from torch.nn import Linear, Module, Sigmoid


class Generator(Module):
    dense_layer: Linear
    activation: Sigmoid
    WEIGHT_FILENAME = "generator.pt"

    def __init__(self: "Generator", length: int):
        super(Generator, self).__init__()
        self.dense_layer = Linear(length, length)
        self.activation = Sigmoid()

    def forward(self: "Generator", x: Tensor) -> Tensor:
        return self.activation(self.dense_layer(x))

    @classmethod
    def load(cls, dir: str) -> "Generator":
        if not dir or not os.path.isdir(dir):
            raise FileNotFoundError("Please provide weights dir")

        path = os.path.join(dir, cls.WEIGHT_FILENAME)
        return torch.load(path)

    def save(self: "Generator", dir: str) -> None:
        if not dir or not os.path.isdir(dir):
            raise FileNotFoundError("Please provide weights dir")

        path = os.path.join(dir, self.WEIGHT_FILENAME)
        torch.save(self, path)
        return
