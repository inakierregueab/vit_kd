import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor, temperature: float = 1., t_is_prob: bool = True) -> torch.Tensor:

        if not t_is_prob:
            target = F.softmax(target/temperature, dim=1)

        loss = torch.sum(-target * F.log_softmax(x/temperature, dim=1), dim=1)
        return loss.mean()