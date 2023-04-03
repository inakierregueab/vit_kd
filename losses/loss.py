import torch.nn.functional as F


def ce_loss(output, target):
    return F.cross_entropy(output, target)