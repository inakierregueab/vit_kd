import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor, temperature: float = 1.) -> torch.Tensor:
        loss = torch.sum(-target/temperature * F.log_softmax(x/temperature, dim=1), dim=1)
        return loss.mean()