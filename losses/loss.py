import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.softCE import SoftTargetCrossEntropy


class DistillationLoss(nn.Module):
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

        assert distillation_type in ['none', 'soft_kl', 'soft_mse', 'soft_ce', 'hard']
        self.distillation_type = distillation_type

        assert distillation_from in ['teacher', 'proxy', 'both']
        self.distillation_from = distillation_from

        self.distillation_alpha = distillation_alpha
        self.distillation_tau = distillation_tau

        if rank == 0:
            print(f'Distillation type is {distillation_type} from {distillation_from} logits,'
                  f' with alpha={distillation_alpha} and tau={distillation_tau}')

        assert hidden_state_criterion in ['none', 'mse', 'cosine']
        self.hidden_state_criterion = hidden_state_criterion
        self.hidden_state_beta = hidden_state_beta

        if rank == 0:
            print(f'Hidden state criterion is {hidden_state_criterion} with beta={hidden_state_beta}')

    def forward(self, outputs, target):
        """
        :param outputs: tuple of (student_output, teacher_output, proxy_output) where each output is a tuple of (logits, hidden_states, attention_weights)
        :param target: logits from target
        :return: loss, base_loss, distill_loss
        """

        # Compute base criterion
        base_loss = self.base_criterion(outputs[0][0], target)

        # Compute distillation loss
        if self.distillation_type == 'none':
             distill_loss = base_loss
        else:
             distill_loss = self.compute_distillation_loss(outputs)

        # Compute hidden state loss
        if self.hidden_state_criterion == 'none':
            hidden_state_loss = base_loss
        else:
            hidden_state_loss = self.compute_hidden_state_loss(outputs)

        # Compute total loss
        total_loss = base_loss * (1-self.distillation_alpha-self.hidden_state_beta) + \
                     distill_loss * self.distillation_alpha + \
                     hidden_state_loss * self.hidden_state_beta

        return total_loss, base_loss, distill_loss, hidden_state_loss

    def compute_distillation_loss(self, outputs):
        # TODO: implement 'both'?
        index = 1 if self.distillation_from == 'teacher' else 2

        if self.distillation_type == 'soft_kl':
            T = self.distillation_tau
            distill_loss = F.kl_div(
                # Use LogSoftmax for numerical stability
                F.log_softmax(outputs[0][0] / T, dim=1),
                F.log_softmax(outputs[index][0] / T, dim=1),
                reduction='batchmean',
                log_target=True
            ) * (T * T)

        elif self.distillation_type == 'soft_mse':
            # More info: https://arxiv.org/pdf/2105.08919.pdf
            distill_loss = F.mse_loss(outputs[0][0], outputs[index][0])

        elif self.distillation_type == 'hard':
            distill_loss = F.cross_entropy(outputs[0][0], outputs[index][0].argmax(dim=1))

        elif self.distillation_type == 'soft_ce':
            distill_loss = SoftTargetCrossEntropy()(outputs[0][0], outputs[index][0],
                                                    temperature=self.distillation_tau, t_is_prob=False)

        return distill_loss

    def compute_hidden_state_loss(self, outputs):

        if self.hidden_state_criterion == 'mse':
            hidden_state_loss = F.mse_loss(outputs[0][1], outputs[2][1])

        elif self.hidden_state_criterion == 'cosine':
            # TODO: along which dimensions should we compute cosine similarity?
            # TODO: Reduction of loss is not clear
            hidden_state_loss = 1 - F.cosine_similarity(outputs[0][1], outputs[2][1], dim=1).mean()

        return hidden_state_loss


class OnlineKDLoss(nn.Module):
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

        assert distillation_type in ['none', 'soft_kl', 'soft_mse', 'soft_ce', 'hard']
        self.distillation_type = distillation_type

        assert distillation_from in ['teacher', 'proxy', 'both']
        self.distillation_from = distillation_from

        self.distillation_alpha = distillation_alpha
        self.distillation_tau = distillation_tau

        if rank == 0:
            print(f'Distillation type is {distillation_type} from {distillation_from} logits,'
                  f' with alpha={distillation_alpha} and tau={distillation_tau}')

        assert hidden_state_criterion in ['none', 'mse', 'cosine']
        self.hidden_state_criterion = hidden_state_criterion
        self.hidden_state_beta = hidden_state_beta

        if rank == 0:
            print(f'Hidden state criterion is {hidden_state_criterion} with beta={hidden_state_beta}')

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
        s_distill_loss = self.compute_distillation_loss(student_logits, teacher_logits)
        p_distill_loss = self.compute_distillation_loss(proxy_logits, teacher_logits)
        # TODO: Proxy logits can be distilled to student logits as well

        # Compute hidden state loss
        hidden_state_loss = self.compute_hidden_state_loss(student_hidden_states, proxy_hidden_states)

        # Student loss
        s_total_loss = s_base_loss * (1-self.distillation_alpha-self.hidden_state_beta) + \
                          s_distill_loss * self.distillation_alpha + \
                            hidden_state_loss * self.hidden_state_beta

        # Proxy loss
        p_total_loss = p_base_loss * (1-self.distillation_alpha-self.hidden_state_beta) + \
                            p_distill_loss * self.distillation_alpha + \
                            hidden_state_loss * self.hidden_state_beta

        # Compute total loss
        # TODO: We can add a weight to the proxy loss
        total_loss = s_total_loss + p_total_loss

        return total_loss, s_base_loss, s_distill_loss, hidden_state_loss

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
