import numpy as np
import torch
import torch.nn as nn
from models.vision_transfromer import VisionTransformer
from models.modified_vit import ModVisionTransformer
from utils.util import load_pretrained_weights


class Teacher_ViTB16(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ModVisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
        )
        self.model = load_pretrained_weights(self.model,'IMAGENET1K_V1')

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)


class DeiT_S16(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
            **kwargs
        )


class ProxyStudent_S16(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
            proxy=True,
            **kwargs
        )


class DeiT_Ti16(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=3,
            hidden_dim=192,
            mlp_dim=768,
            **kwargs
        )


class TandemTPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher = Teacher_ViTB16()
        self.proxy_student = ProxyStudent_S16()

    def forward(self, x):
        with torch.no_grad():
            t_output = self.teacher(x)

        s_output = self.proxy_student(x, t_output[1], output_hidden=True)   #TODO: output_hidden=True only in inference
        return s_output, t_output


class TandemTPS_noTT(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher = Teacher_ViTB16()
        self.proxy_student = ProxyStudent_S16()
        self.dummy_token = nn.Parameter(torch.zeros(1, 1, 768))

    def forward(self, x):
        with torch.no_grad():
            t_output = self.teacher(x)
            # TODO: seq_length of teacher is reduced by 1, thus attention matrices of proxy aren't square (bad if we want to use them for distillation)
            t_hidden_state = t_output[1][:, 1:, :]

        # TODO: train an auxiliary token?
        n = x.shape[0]
        dummy_token = self.dummy_token.expand(n, -1, -1)
        t_hidden_state = torch.cat((dummy_token, t_hidden_state), dim=1)

        s_output = self.proxy_student(x, t_hidden_state, output_hidden=False, output_att=True, average_att=True)
        return s_output, t_output


class TandemPSS(nn.Module):
    def __init__(self):
        super().__init__()
        self.proxy = TandemTPS()
        checkpoint = torch.load('./../../saved/weights/proxy_kl/checkpoint-epoch60.pth',
                                map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.proxy.load_state_dict(checkpoint['state_dict'])
        for param in self.proxy.parameters():
            param.requires_grad = False

        self.student = DeiT_S16()

    def forward(self, x):
        with torch.no_grad():
            p_output, t_output = self.proxy(x)

        s_output = self.student(x, output_hidden=True)
        return s_output, t_output, p_output


class OnlinePSS(nn.Module):
    def __init__(self):
        super().__init__()
        self.teacher = Teacher_ViTB16()
        self.proxy = ProxyStudent_S16()
        self.student = DeiT_S16()

        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            t_output = self.teacher(x)

        p_output = self.proxy(x, t_output[1], output_hidden=True)
        s_output = self.student(x, output_hidden=True)
        return s_output, t_output, p_output


# Testing unit
if __name__ == "__main__":

    # Parameters
    bs = 1
    image_size = 224
    patch_size = 16
    t_hidden_dim = 768
    s_hidden_dim = 384
    num_classes = 1000
    seq_length = (image_size // patch_size) ** 2 + 1  # Class token

    # Random inputs
    x = torch.rand(bs, 3, image_size, image_size)
    memory = torch.rand(bs, seq_length, t_hidden_dim)

    # Model checks
    teacher = Teacher_ViTB16()
    out = teacher(x)

    # 1. Teacher outputs (cls_token, x, A)
    assert len(out) == 3
    # 2. cls_token is a tensor of shape (bs, num_classes)
    assert out[0].shape == (bs, num_classes)
    # 3. x is a tensor of shape (bs, seq_length, hidden_dim)
    assert out[1].shape == (bs, seq_length, t_hidden_dim)
    # 4. A is None
    assert out[2] is None
    # 4. No params requiring grad
    assert len([True for p in teacher.parameters() if p.requires_grad]) == 0

    tandem = TandemTPS()
    s_out, t_out = tandem(x)

    # 5. Tandem outputs both student and teacher correctly
    assert s_out[0].shape == (bs, num_classes)
    assert t_out[0].shape == (bs, num_classes)

    # 6. Tandem trainable params are the same as proxy student
    tandem_parameters = filter(lambda p: p.requires_grad, tandem.parameters())
    tandem_params = sum([np.prod(p.size()) for p in tandem_parameters])

    proxy_parameters = filter(lambda p: p.requires_grad, tandem.proxy_student.parameters())
    proxy_params = sum([np.prod(p.size()) for p in proxy_parameters])

    assert tandem_params == proxy_params

    # Triplet tandem
    triplet = TandemPSS()
    s_out, t_out, p_out = triplet(x)

    # 7. Logits have same shape
    assert s_out[0].shape == t_out[0].shape == p_out[0].shape == (bs, num_classes)

    # 8. Hidden states have same shape
    assert s_out[1].shape == p_out[1].shape == (bs, seq_length, s_hidden_dim)

    # 9. Online tandem has same number of trainable params as proxy plus student
    online = OnlinePSS()
    online_parameters = filter(lambda p: p.requires_grad, online.parameters())
    online_params = sum([np.prod(p.size()) for p in online_parameters])

    proxy_parameters = filter(lambda p: p.requires_grad, online.proxy.parameters())
    proxy_params = sum([np.prod(p.size()) for p in proxy_parameters])

    student_parameters = filter(lambda p: p.requires_grad, online.student.parameters())
    student_params = sum([np.prod(p.size()) for p in student_parameters])

    assert online_params == proxy_params + student_params





