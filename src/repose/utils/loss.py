from dataclasses import dataclass

import torch
from torch.functional import Tensor
from torch.nn import BCEWithLogitsLoss


@dataclass
class Loss:
    @staticmethod
    def calc(tensor: Tensor, smooth: bool = False, real: bool = True) -> Tensor:
        labels = Loss.__labels(tensor, real=real) * (0.9 if smooth else 1.0)
        criterion = BCEWithLogitsLoss()
        return criterion(tensor.squeeze(), labels)  # calculate loss

    @staticmethod
    def __labels(tensor: Tensor, real: bool = True) -> Tensor:
        batch_size = tensor.size(0)
        return torch.ones(batch_size) if real else torch.zeros(batch_size)
