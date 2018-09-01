"""Modules for building the char rnn"""

import torch
from torch import nn
from torch import optim
import torch.functional as F

from pytorch_utils.wrapped_lstm import WrappedLSTM

class Enc(nn.Module):
    """"""
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
