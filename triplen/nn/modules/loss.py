from triplen.nn.modules.module import Module
from .. import functional as F


class CrossEntropyLoss(Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        return F.cross_entropy_loss(input, target)

