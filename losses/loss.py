from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.softCE import SoftTargetCrossEntropy


class BaseKDloss(nn.Module):
    def __init__(self,
                 distillation_type='none',
                 distillation_from='teacher',
                 distillation_alpha=0,
                 distillation_tau=1,
                 hidden_state_criterion='none',
                 hidden_state_beta=0,
                 rank=0):
        super().__init__()

        self.base_criterion = SoftTargetCrossEntropy()

        assert distillation_type in ['soft_kl', 'soft_mse', 'soft_ce', 'hard']
        self.distillation_type = distillation_type

        assert distillation_from in ['teacher', 'proxy', 'both']
        self.distillation_from = distillation_from

        self.distillation_alpha = distillation_alpha
        self.distillation_tau = distillation_tau

        if rank == 0:
            print(f'Distillation type is {distillation_type} from {distillation_from} logits,'
                  f' with alpha={distillation_alpha} and tau={distillation_tau}')

        assert hidden_state_criterion in ['mse', 'cosine']
        self.hidden_state_criterion = hidden_state_criterion
        self.hidden_state_beta = hidden_state_beta

        if rank == 0:
            print(f'Hidden state criterion is {hidden_state_criterion} with beta={hidden_state_beta}')

    def compute_distillation_loss(self, student_logits, teacher_logits):

        if self.distillation_type == 'soft_kl':
            T = self.distillation_tau
            distill_loss = F.kl_div(
                # Use LogSoftmax for numerical stability
                F.log_softmax(student_logits / T, dim=1),
                F.log_softmax(teacher_logits / T, dim=1),
                reduction='batchmean',
                log_target=True
            ) * (T * T)

        elif self.distillation_type == 'soft_mse':
            # More info: https://arxiv.org/pdf/2105.08919.pdf
            distill_loss = F.mse_loss(student_logits, teacher_logits)

        elif self.distillation_type == 'hard':
            distill_loss = F.cross_entropy(student_logits, teacher_logits.argmax(dim=1))

        elif self.distillation_type == 'soft_ce':
            distill_loss = SoftTargetCrossEntropy()(student_logits, teacher_logits,
                                                    temperature=self.distillation_tau, t_is_prob=False)

        return distill_loss

    def compute_hidden_state_loss(self, student_hidden_states, teacher_hidden_states):

        if self.hidden_state_criterion == 'mse':
            hidden_state_loss = F.mse_loss(student_hidden_states, teacher_hidden_states)

        elif self.hidden_state_criterion == 'cosine':
            hidden_state_loss = 1 - F.cosine_similarity(student_hidden_states, teacher_hidden_states, dim=1).mean()

        return hidden_state_loss

    @abstractmethod
    def forward(self, outputs, target):
        raise NotImplementedError


class ProxyKDLoss(BaseKDloss):
    def __init__(self,
                 distillation_type='none',
                 distillation_from='teacher',
                 distillation_alpha=0,
                 distillation_tau=1,
                 hidden_state_criterion='none',
                 hidden_state_beta=0,
                 rank=0):
        super().__init__(distillation_type=distillation_type,
              distillation_from=distillation_from,
              distillation_alpha=distillation_alpha,
              distillation_tau=distillation_tau,
              hidden_state_criterion=hidden_state_criterion,
              hidden_state_beta=hidden_state_beta,
              rank=rank)

    def forward(self, outputs, target):
        """
                :param outputs: tuple of (student_output, teacher_output, proxy_output) where each output is a tuple of (logits, hidden_states, attention_weights)
                :param target: logits from target
                :return: loss, base_loss, distill_loss
                """

        # Unpack outputs
        proxy_logits, proxy_hidden_states, proxy_attention_weights = outputs[0]
        teacher_logits, teacher_hidden_states, teacher_attention_weights = outputs[1]


        # Compute base criterion
        base_loss = self.base_criterion(proxy_logits, target)

        # Compute distillation loss
        distill_loss = self.compute_distillation_loss(proxy_logits, teacher_logits)

        # Compute total loss
        total_loss = base_loss * (1 - self.distillation_alpha) + distill_loss * self.distillation_alpha

        return total_loss, base_loss, distill_loss, 0


class OfflineKDLoss(BaseKDloss):
    def __init__(self,
                 distillation_type='none',
                 distillation_from='teacher',
                 distillation_alpha=0,
                 distillation_tau=1,
                 hidden_state_criterion='none',
                 hidden_state_beta=0,
                 rank=0):
        super().__init__(distillation_type=distillation_type,
                         distillation_from=distillation_from,
                         distillation_alpha=distillation_alpha,
                         distillation_tau=distillation_tau,
                         hidden_state_criterion=hidden_state_criterion,
                         hidden_state_beta=hidden_state_beta,
                         rank=rank)

    def forward(self, outputs, target):
        """
        :param outputs: tuple of (student_output, teacher_output, proxy_output) where each output is a tuple of (logits, hidden_states, attention_weights)
        :param target: logits from target
        :return: loss, base_loss, distill_loss
        """

        # Unpack outputs
        student_logits, student_hidden_states, student_attention_weights = outputs[0]
        teacher_logits, teacher_hidden_states, teacher_attention_weights = outputs[1]
        proxy_logits, proxy_hidden_states, proxy_attention_weights = outputs[2]

        # Compute base criterion
        base_loss = self.base_criterion(student_logits, target)

        # Compute distillation loss
        distill_loss = self.compute_distillation_loss(student_logits, teacher_logits)

        # Compute hidden state loss
        hidden_state_loss = self.compute_hidden_state_loss(student_hidden_states, proxy_hidden_states)

        # Compute total loss
        total_loss = base_loss * (1-self.distillation_alpha-self.hidden_state_beta) + \
                     distill_loss * self.distillation_alpha + \
                     hidden_state_loss * self.hidden_state_beta

        return total_loss, base_loss, distill_loss, hidden_state_loss


class OnlineKDLoss(nn.Module):
    def __init__(self, logits_criterion='none', hidden_state_criterion='none', s_gamma=0, s_alpha=0, s_tau=1, s_beta=0,
                 p_gamma=0, p_alpha=0, p_tau=1, p_beta=0, rank=0):

        super().__init__()
        self.logits_criterion = logits_criterion
        self.hidden_state_criterion = hidden_state_criterion

        self.s_gamma = s_gamma
        self.s_alpha = s_alpha
        self.s_tau = s_tau
        self.s_beta = s_beta

        self.p_gamma = p_gamma
        self.p_alpha = p_alpha
        self.p_tau = p_tau
        self.p_beta = p_beta

        self.base_criterion = SoftTargetCrossEntropy()

    def compute_distillation_loss(self, student_logits, teacher_logits, T=1):

        if self.logits_criterion == 'soft_kl':
            distill_loss = F.kl_div(
                # Use LogSoftmax for numerical stability
                F.log_softmax(student_logits / T, dim=1),
                F.log_softmax(teacher_logits / T, dim=1),
                reduction='batchmean',
                log_target=True
            ) * (T * T)

        elif self.logits_criterion == 'soft_mse':
            # More info: https://arxiv.org/pdf/2105.08919.pdf
            distill_loss = F.mse_loss(student_logits, teacher_logits)

        elif self.logits_criterion == 'hard':
            distill_loss = F.cross_entropy(student_logits, teacher_logits.argmax(dim=1))

        elif self.logits_criterion == 'soft_ce':
            distill_loss = SoftTargetCrossEntropy()(student_logits, teacher_logits,
                                                    temperature=T, t_is_prob=False)

        return distill_loss

    def compute_hidden_state_loss(self, student_hidden_states, teacher_hidden_states):

        if self.hidden_state_criterion == 'mse':
            hidden_state_loss = F.mse_loss(student_hidden_states, teacher_hidden_states)

        elif self.hidden_state_criterion == 'cosine':
            hidden_state_loss = 1 - F.cosine_similarity(student_hidden_states, teacher_hidden_states, dim=1).mean()

        return hidden_state_loss

    def forward(self, outputs, target):
        """
        :param outputs: tuple of (student_output, teacher_output, proxy_output) where each output is a tuple of (logits, hidden_states, attention_weights)
        :param target: logits from target
        :return: loss, base_loss, distill_loss
        """

        # Unpack outputs
        student_logits, student_hidden_states, student_attention_weights = outputs[0]
        teacher_logits, teacher_hidden_states, teacher_attention_weights = outputs[1]
        proxy_logits, proxy_hidden_states, proxy_attention_weights = outputs[2]

        # Compute base criterion
        s_base_loss = self.base_criterion(student_logits, target)
        p_base_loss = self.base_criterion(proxy_logits, target)

        # Compute distillation loss
        s_distill_loss = self.compute_distillation_loss(student_logits, teacher_logits, T=self.s_tau)
        p_distill_loss = self.compute_distillation_loss(proxy_logits, teacher_logits, T=self.p_tau)

        # Compute hidden state loss
        hidden_state_loss = self.compute_hidden_state_loss(student_hidden_states, proxy_hidden_states)

        # Student loss
        s_total_loss = s_base_loss * self.s_gamma + s_distill_loss * self.s_alpha + hidden_state_loss * self.s_beta

        # Proxy loss
        p_total_loss = p_base_loss * self.p_gamma + p_distill_loss * self.p_alpha + hidden_state_loss * self.p_beta

        # Compute total loss
        # TODO: We can add a weight to the proxy loss
        total_loss = s_total_loss + p_total_loss

        return total_loss, s_base_loss, s_distill_loss, hidden_state_loss

class StudentKDLoss(BaseKDloss):
    def __init__(self,
                 distillation_type='none',
                 distillation_from='teacher',
                 distillation_alpha=0,
                 distillation_tau=1,
                 hidden_state_criterion='none',
                 hidden_state_beta=0,
                 rank=0):
        super().__init__(distillation_type=distillation_type,
                         distillation_from=distillation_from,
                         distillation_alpha=distillation_alpha,
                         distillation_tau=distillation_tau,
                         hidden_state_criterion=hidden_state_criterion,
                         hidden_state_beta=hidden_state_beta,
                         rank=rank)

    def forward(self, outputs, target):
        """
        :param outputs: tuple of (student_output, teacher_output, proxy_output) where each output is a tuple of (logits, hidden_states, attention_weights)
        :param target: logits from target
        :return: loss, base_loss, distill_loss
        """

        # Unpack outputs
        student_logits, student_hidden_states, student_attention_weights = outputs[0]
        teacher_logits, teacher_hidden_states, teacher_attention_weights = outputs[1]

        # Compute base criterion
        s_base_loss = self.base_criterion(student_logits, target)

        # Compute distillation loss
        s_distill_loss = self.compute_distillation_loss(student_logits, teacher_logits)

        # Compute hidden state loss
        hidden_state_loss = self.compute_hidden_state_loss(student_hidden_states, teacher_hidden_states)

        # Student loss
        total_loss = s_base_loss * (1-self.distillation_alpha-self.hidden_state_beta) + \
                          s_distill_loss * self.distillation_alpha + \
                            hidden_state_loss * self.hidden_state_beta

        return total_loss, s_base_loss, s_distill_loss, hidden_state_loss
