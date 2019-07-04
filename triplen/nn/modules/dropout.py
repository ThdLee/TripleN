import numpy as np
from triplen.nn.modules.module import Module
from .. import functional as F


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, input):
        return F.dropout(input, self.p, self.training)