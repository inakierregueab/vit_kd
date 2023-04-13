import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def ce_loss(output, target):
    return F.cross_entropy(output, target)


class DistillationLoss(nn.Module):
    def __init__(self, base_criterion=F.cross_entropy, distillation_type='none', alpha=0, tau=1, rank=0):
        super().__init__()
        # TODO: base critarion switch to softCE when using label smoothing
        self.base_criterion = base_criterion
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        if rank == 0:
            print(f'Distillation type is {distillation_type}')

        self.alpha = alpha
        self.tau = tau

    def forward(self, outputs, target):

        if not isinstance(outputs, torch.Tensor):
            s_output, t_output = outputs
        else:
            s_output = outputs

        base_loss = self.base_criterion(s_output, target)

        if self.distillation_type == 'none':
            return base_loss

        elif self.distillation_type == 'soft':
            T = self.tau
            distill_loss = F.kl_div(
                # Use LogSoftmax for numerical stability
                F.log_softmax(s_output/T, dim=1),
                F.log_softmax(t_output/T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T*T)/s_output.numel()

        elif self.distillation_type == 'hard':
            distill_loss = F.cross_entropy(s_output, t_output.argmax(dim=1))

        loss = base_loss * (1-self.alpha) + distill_loss * self.alpha
        return loss




