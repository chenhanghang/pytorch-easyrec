import torch.nn.functional as F
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)


def bce_loss(output, target):
    #return F.binary_cross_entropy(output, target)
    criterion = nn.BCELoss()
    return criterion(output, target)