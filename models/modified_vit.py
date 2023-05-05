import torch
from torchvision.models import VisionTransformer

from utils.util import load_pretrained_weights


class ModVisionTransformer(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        cls_token = x[:, 0]
        cls_token = self.heads(cls_token)

        return cls_token, x

# Testing unit
if __name__ == "__main__":

    # Parameters
    bs = 1
    image_size = 224
    patch_size = 16
    t_hidden_dim = 768
    s_hidden_dim = 384
    num_classes = 1000
    seq_length = (image_size // patch_size) ** 2 + 1    # Class token

    # Random inputs
    x = torch.rand(bs, 3, image_size, image_size)
    memory = torch.rand(bs, seq_length, t_hidden_dim)

    # Model checks:
    teacher = ModVisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=12,
        num_heads=12,
        hidden_dim=t_hidden_dim,
        mlp_dim=3072,
    )
    teacher = load_pretrained_weights(teacher,'IMAGENET1K_V1')

    out = teacher(x)

    # 1. Teacher outputs (cls_token, x)
    assert len(out) == 2
    # 2. cls_token is a tensor of shape (bs, num_classes)
    assert out[0].shape == (bs, num_classes)
    # 3. x is a tensor of shape (bs, seq_length, hidden_dim)
    assert out[1].shape == (bs, seq_length, t_hidden_dim)