import torch
import torch.nn as nn
import torch.nn.functional as F


def ce_loss(output, target):
    return F.cross_entropy(output, target)

# TODO: test it
class DistillationLoss(nn.Module):
    def __init__(
            self,
            base_criterion: nn.Module,
            distillation_type: str,
            alpha: float,
            tau: float
    ):
        super().__init__()
        self.base_criterion = base_criterion
        assert distillation_type in ['soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, output, target, teacher_output):
        base_loss = self.base_criterion(output, target)

        if self.distillation_type == 'soft':
            T = self.tau
            distill_loss = F.kl_div(
                # Use LogSoftmax for numerical stability
                F.log_softmax(output/T, dim=1),
                F.log_softmax(teacher_output/T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T*T)/output.numel()

        elif self.distillation_type == 'hard':
            distill_loss = F.cross_entropy(output, teacher_output.argmax(dim=1))

        loss = base_loss * (1-self.alpha) + distill_loss * self.alpha
        return loss




