import math
import numpy as np

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from torchvision.models.vision_transformer import MLPBlock


class ProxyEncoderBlock(nn.Module):
    """Proxy Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
            kdim=768,
            vdim=768,
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self,
                input: torch.Tensor,
                memory: torch.Tensor,
                output_att: bool = False,
                average_att: bool = False,
                ):

        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = input
        x = self.ln_1(x)
        x, A = self.cross_attention(query=x, key=memory, value=memory, need_weights=output_att, average_attn_weights=average_att)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        x = x + y

        return x, A


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self,
                input: torch.Tensor,
                output_att: bool = False,
                average_att: bool = False):

        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, A = self.self_attention(x, x, x, need_weights=output_att, average_attn_weights=average_att)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        x = x + y
        return x, A


class modEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
            self,
            seq_length: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            proxy: bool = False,
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.num_layers = num_layers
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        self.proxy = proxy
        encoder_block = EncoderBlock if not proxy else ProxyEncoderBlock

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(encoder_block(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            ))
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                output_att: bool = False,
                average_att: bool = False,
                ):

        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        x = self.dropout(input)

        if self.proxy is False:
            for layer in self.layers:
                x, A = layer(x, output_att=output_att, average_att=average_att)
        else:
            for layer in self.layers:
                x, A = layer(x, memory=memory, output_att=output_att, average_att=average_att)

        x = self.ln(x)
        return x, A


class CSPEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
            self,
            seq_length: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float,
            attention_dropout: float,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.num_layers = num_layers
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)
        cross_block = ProxyEncoderBlock
        self_block = EncoderBlock

        self.layers = nn.ModuleList()

        self.layers.append(self_block(
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        ))

        self.layers.append(cross_block(
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        ))

        self.layers.append(self_block(
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        ))

        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                output_att: bool = False,
                average_att: bool = False,
                ):

        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        x = self.dropout(input)

        for layer in self.layers:
            if isinstance(layer, ProxyEncoderBlock):
                x, _ = layer(x, memory=memory, output_att=output_att, average_att=average_att)
            elif isinstance(layer, EncoderBlock):
                x, A = layer(x, output_att=output_att, average_att=average_att)

        x = self.ln(x)
        return x, A


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_classes: int = 1000,
            representation_size: Optional[int] = None,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            proxy: bool = False,
            self_proxy: bool = False,
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        if self_proxy is False:
            self.encoder = modEncoder(
                seq_length,
                num_layers,
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
                proxy
            )
        else:
            self.encoder = CSPEncoder(
                seq_length,
                num_layers,
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer
            )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor,
                memory: Optional[torch.Tensor] = None,
                output_hidden: bool = False,
                output_att: bool = False,
                average_att: bool = False
                ):
        """
        Arguments:
            x: torch.Tensor - Input tensor of shape (batch, channels, height, width)
            memory: Optional[torch.Tensor] - Memory tensor of shape (batch, seq_len, t_hidden_dim)
            output_hidden: bool - Whether to output the last hidden state (batch, seq_len, hidden_dim)
            output_att: bool - Whether to output the last attention matrices (batch, num_heads, seq_len, seq_len)
            average_att: bool - Whether to average across heads the last attention matrices (batch, seq_len, seq_len)
        Returns:
            cls_token: torch.Tensor - Tensor of shape (N, classes)
            x: torch.Tensor - Tensor of shape (N, seq_len, hidden_dim) or None
            A: torch.Tensor - Tensor of shape (N, num_heads, seq_len, seq_len) or (N, seq_len, seq_len) or None
        """

        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x, A = self.encoder(x, memory=memory, output_att=output_att, average_att=average_att)

        # Classifier "token" as used by standard language architectures
        cls_token = x[:, 0]
        cls_token = self.heads(cls_token)

        return (cls_token, x, A) if output_hidden else (cls_token, None, A)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


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
    num_heads = 12

    # Random inputs
    x = torch.rand(bs, 3, image_size, image_size)
    memory = torch.rand(bs, seq_length, t_hidden_dim)

    # Model checks:
    teacher = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=12,
        num_heads=num_heads,
        hidden_dim=t_hidden_dim,
        mlp_dim=3072,
    )

    # Only class token
    out = teacher(x, output_hidden=False, output_att=False, average_att=False)
    # 1. Vit outputs a tuple of length 3
    assert len(out) == 3
    # 2. cls_token is a tensor of shape (bs, num_classes)
    assert out[0].shape == (bs, num_classes)
    # 3. The last hidden state and attention matrices are None
    assert out[1] is None
    assert out[2] is None

    # Outputs hidden state and attention matrices
    out = teacher(x, output_hidden=True, output_att=True, average_att=False)
    # 4. x is a tensor of shape (bs, seq_length, hidden_dim)
    assert out[1].shape == (bs, seq_length, t_hidden_dim)
    # 5. A is a tensor of shape (bs, num_heads, seq_length, seq_length)
    assert out[2].shape == (bs, num_heads, seq_length, seq_length)

    # Average attention matrices
    t_out = teacher(x, output_hidden=True, output_att=True, average_att=True)
    # 6. A is a tensor of shape (bs, seq_length, seq_length)
    assert t_out[2].shape == (bs, seq_length, seq_length)

    proxy_student = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=12,
        num_heads=6,
        hidden_dim=s_hidden_dim,
        mlp_dim=1536,
        proxy=True
    )
    out = proxy_student(x, t_out[1], output_hidden=True, output_att=True, average_att=True)

    # 7. Proxy student outputs class token of shape (bs, num_classes)
    assert out[0].shape == (bs, num_classes)
    # 8. Proxy student outputs hidden state of shape (bs, seq_length, hidden_dim)
    assert out[1].shape == (bs, seq_length, s_hidden_dim)
    # 9. Proxy student outputs averaged attention matrices of shape (bs, seq_length, seq_length)
    assert out[2].shape == (bs, seq_length, seq_length)

    self_proxy_student = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=12,
        num_heads=6,
        hidden_dim=s_hidden_dim,
        mlp_dim=1536,
        proxy=True,
        self_proxy=True
    )
    out = self_proxy_student(x, t_out[1], output_hidden=True, output_att=True, average_att=True)

    # 10. Self proxy student outputs class token of shape (bs, num_classes)
    assert out[0].shape == (bs, num_classes)
    # 11. Self proxy student outputs hidden state of shape (bs, seq_length, hidden_dim)
    assert out[1].shape == (bs, seq_length, s_hidden_dim)
    # 12. Self proxy student outputs averaged attention matrices of shape (bs, seq_length, seq_length)
    assert out[2].shape == (bs, seq_length, seq_length)



