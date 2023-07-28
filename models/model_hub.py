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


class SelfProxyStudent_S16(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            image_size=64,
            patch_size=8,
            num_layers=12,
            num_heads=3,
            hidden_dim=192,
            mlp_dim=768,
            proxy=True,
            self_proxy=True,
            **kwargs
        )


class DeiT_Ti16(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            image_size=64,
            patch_size=8,
            num_layers=12,
            num_heads=3,
            hidden_dim=192,
            mlp_dim=768,
            **kwargs
        )


class TP(nn.Module):
    """PROXY STUDENT + TEACHER"""
    def __init__(self):
        super().__init__()
        self.teacher = Teacher_ViTB16()
        self.proxy_student = SelfProxyStudent_S16()

    def forward(self, x):
        with torch.no_grad():
            t_output = self.teacher(x)

        #s_output = self.proxy_student(x, t_output[1], output_hidden=True, output_att=True, average_att=True)
        s_output = self.proxy_student(x, t_output[1], output_hidden=True, should_resize=True)
        return s_output, t_output, 0


class TPS_offline(nn.Module):
    """PROXY STUDENT + STUDENT + TEACHER learning offline"""
    def __init__(self):
        super().__init__()
        self.teacher_proxy = TP()
        checkpoint = torch.load('./../../saved/weights/proxy_kl/checkpoint-epoch60.pth',
                                map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.teacher_proxy.load_state_dict(checkpoint['state_dict'])
        for param in self.teacher_proxy.parameters():
            param.requires_grad = False

        self.student = DeiT_S16()

    def forward(self, x):
        with torch.no_grad():
            p_output, t_output, _ = self.teacher_proxy(x)

        s_output = self.student(x, output_hidden=True, should_resize=True)
        return s_output, t_output, p_output


class TPS_online(nn.Module):
    """PROXY STUDENT + STUDENT + TEACHER learning online"""
    def __init__(self):
        super().__init__()
        self.teacher = Teacher_ViTB16()
        self.proxy = SelfProxyStudent_S16() #ProxyStudent_S16()
        self.student = DeiT_Ti16()

        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            t_output = self.teacher(x)

        p_output = self.proxy(x, t_output[1], output_hidden=True, should_resize=True)
        s_output = self.student(x, output_hidden=True, should_resize=True)
        return s_output, t_output, p_output


class TS(nn.Module):
    """STUDENT + TEACHER"""
    def __init__(self):
        super().__init__()
        self.teacher = Teacher_ViTB16()
        self.student = DeiT_Ti16()
        self.mlp = nn.Linear(in_features=192, out_features=768)
        self.mlp2 = nn.Linear(in_features=self.student.seq_length, out_features=self.teacher.model.seq_length)

    def forward(self, x):
        with torch.no_grad():
            t_output = self.teacher(x)

        cls_token, hidden, matrix = self.student(x, output_hidden=True, should_resize=True)
        hidden = self.mlp(hidden)
        hidden = self.mlp2(hidden.permute(0, 2, 1)).permute(0, 2, 1)
        return (cls_token, hidden, matrix), t_output, 0


# Testing unit
if __name__ == "__main__":

    # Parameters
    bs = 1
    image_size = 224
    patch_size = 16
    t_hidden_dim = 768
    s_hidden_dim = 192
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

    tandem = TP()
    s_out, t_out, _ = tandem(x)

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
    #triplet = TandemPSS()
    #s_out, t_out, p_out = triplet(x)

    # 7. Logits have same shape
    #assert s_out[0].shape == t_out[0].shape == p_out[0].shape == (bs, num_classes)

    # 8. Hidden states have same shape
    #assert s_out[1].shape == p_out[1].shape == (bs, seq_length, s_hidden_dim)

    # 9. Online tandem has same number of trainable params as proxy plus student
    online = TPS_online()
    online_parameters = filter(lambda p: p.requires_grad, online.parameters())
    online_params = sum([np.prod(p.size()) for p in online_parameters])

    proxy_parameters = filter(lambda p: p.requires_grad, online.proxy.parameters())
    proxy_params = sum([np.prod(p.size()) for p in proxy_parameters])

    student_parameters = filter(lambda p: p.requires_grad, online.student.parameters())
    student_params = sum([np.prod(p.size()) for p in student_parameters])

    assert online_params == proxy_params + student_params

    self_proxy = SelfProxyStudent_S16()
    s_out = self_proxy(x, memory, output_hidden=True, output_att=True, average_att=True, should_resize=True)

    # 10. Self proxy outputs (cls_token, x, A)
    assert len(s_out) == 3
    # 11. cls_token is a tensor of shape (bs, num_classes)
    assert s_out[0].shape == (bs, num_classes)
    # 12. x is a tensor of shape (bs, seq_length, hidden_dim)
    assert s_out[1].shape == (bs, 65, s_hidden_dim)
    # 13. A is a tensor of shape (bs, seq_length, seq_length)
    assert s_out[2].shape == (bs, 65, 65)
    # 14. Less params requiring grad than proxy student but more than student
    self_proxy_parameters = filter(lambda p: p.requires_grad, self_proxy.parameters())
    self_proxy_params = sum([np.prod(p.size()) for p in self_proxy_parameters])

   #assert student_params < self_proxy_params < proxy_params

    ts = TS()
    s_out, t_out, _ = ts(x)

    # 15. TS student hidden state is a tensor of same shape as teacher hidden state
    assert s_out[1].shape == t_out[1].shape





