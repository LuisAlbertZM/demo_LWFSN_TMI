import numpy as np

import torch
import torch.nn as nn
from torch.nn import init

class convKernel(nn.Module):
    def __init__(self, out_channels, prev_channels, sx, sy):
        super(convKernel, self).__init__()
        self.kern = nn.Parameter( torch.rand((out_channels, prev_channels, sx, sy)))
        n = prev_channels*sx*sy
        stdv = 1. / np.sqrt(n)
        self.kern.data.uniform_(-stdv, stdv)
        
        self.register_parameter("kern", self.kern)
    def forward(self ):
        return(self.kern)