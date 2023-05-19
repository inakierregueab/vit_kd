import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.softCE import SoftTargetCrossEntropy


class DistillationLoss(nn.Module):
    def __init__(self, distillation_type='none', alpha=0, tau=1, rank=0):
        super().__init__()
        self.base_criterion = SoftTargetCrossEntropy()
        assert distillation_type in ['none', 'soft_kl', 'soft_mse', 'soft_ce', 'hard']
        self.distillation_type = distillation_type
        if rank == 0:
            print(f'Distillation type is {distillation_type}')

        self.alpha = alpha
        self.tau = tau

    def forward(self, outputs, target):
        """
        :param outputs: tuple of (student_output, teacher_output) where each output is a tuple of (logits, hidden_states, attention_weights)
        :param target: logits from target
        :return: loss, base_loss, distill_loss
        """

        s_output, t_output = outputs

        base_loss = self.base_criterion(s_output[0], target)

        if self.distillation_type == 'none':
            return base_loss, base_loss, base_loss

        elif self.distillation_type == 'soft_kl':
            T = self.tau
            distill_loss = F.kl_div(
                # Use LogSoftmax for numerical stability
                F.log_softmax(s_output[0]/T, dim=1),
                F.log_softmax(t_output[0]/T, dim=1),
                reduction='batchmean',
                log_target=True
            ) * (T*T)

        elif self.distillation_type == 'soft_mse':
            # More info: https://arxiv.org/pdf/2105.08919.pdf
            distill_loss = F.mse_loss(s_output[0], t_output[0])

        elif self.distillation_type == 'hard':
            distill_loss = F.cross_entropy(s_output[0], t_output[0].argmax(dim=1))

        elif self.distillation_type == 'soft_ce':
            distill_loss = SoftTargetCrossEntropy()(s_output[0], t_output[0], temperature=self.tau)

        loss = base_loss * (1-self.alpha) + distill_loss * self.alpha
        return loss, base_loss, distill_loss




