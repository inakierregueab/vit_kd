import torch
import torch.nn as nn
from models.vision_transfromer import VisionTransformer



class ViTB16(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            **kwargs
        )


class Teacher_ViTB16(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.model = ViTB16(as_teacher=True)

        checkpoint_path = str(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        # Load architecture params from checkpoint.
        if checkpoint['config']['arch']['type'] != self.model.__class__.__name__:
            print("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

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


class Student_Ti16(VisionTransformer):
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
        self.teacher = Teacher_ViTB16(checkpoint_path='./../saved/models/ViTB16/0404_202607/model_best.pth')
        self.proxy_student = ProxyStudent_S16()

    def forward(self, x):
        with torch.no_grad():
            output_teacher, hidden_state = self.teacher(x)

        output = self.proxy_student(x, hidden_state)
        return output, output_teacher




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
    teacher = Teacher_ViTB16(checkpoint_path='./../saved/models/ViTB16/0404_202607/model_best.pth')
    out = teacher(x)

    # 1. Teacher outputs (cls_token, x)
    assert len(out) == 2
    # 2. cls_token is a tensor of shape (bs, num_classes)
    assert out[0].shape == (bs, num_classes)
    # 3. x is a tensor of shape (bs, seq_length, hidden_dim)
    assert out[1].shape == (bs, seq_length, t_hidden_dim)
    # 4. No params requiring grad
    assert len([True for p in teacher.parameters() if p.requires_grad]) == 0
    # TODO: check performance of teacher, are weights uploaded correctly?

    tandem = TandemTPS()
    s_out, t_out = tandem(x)

    # 5. Tandem outputs both student and teacher correctly
    assert s_out.shape == (bs, num_classes)
    assert t_out.shape == (bs, num_classes)






