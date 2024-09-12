import torch
import torch.nn as nn

class Chomp(nn.Module):
    """
    Removes the last s elements of a time series. 
    Given the three dimensional input (B, C, L) returns (B, C, L - s)
    
    -------------------
    Attributes:
        s: int
            Number of elements to remove
    ------------------
    """

    def __init__(self, s) -> None:
        super().__init__()
        self.s = s

    def forward(self, x):
        return x[:, :, :-self.s]
    


class SqueezeChannels(nn.Module):
    """
    Squeezes the third dimension of a three-dimensional vector
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.squeeze(2)


def get_activation_func(code):
    """
    parser for activation function 
    0 -> LeakyRelu
    """
    if code == 0:
        return nn.LeakyReLU()