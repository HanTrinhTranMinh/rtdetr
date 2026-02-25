# Chứa ConvNeXt (timm)
# models/backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm2d(nn.Module):
    """LayerNorm theo channel cho tensor N,C,H,W (giống ConvNeXt)."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (N,C,H,W)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0.0, layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma[:, None, None] * x
        x = residual + self.drop_path(x)
        return x

class ConvNeXtTiny(nn.Module):
    """
    Trả ra 3 feature maps: C3, C4, C5 (stride 8,16,32) dùng cho detector.
    """
    def __init__(self, in_chans=3, depths=(3,3,9,3), dims=(96,192,384,768), drop_path_rate=0.1):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        # stem
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),  # stride 4
            LayerNorm2d(dims[0]),
        )
        self.downsample_layers.append(stem)
        # 3 downsample tiếp
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                )
            )

        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(ConvNeXtBlock(dims[i], drop_path=dp_rates[cur + j]))
            cur += depths[i]
            self.stages.append(nn.Sequential(*blocks))

        self.out_channels = {"c3": dims[1], "c4": dims[2], "c5": dims[3]}

    def forward(self, x):
        # stage0 (stride4)
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        # stage1 (stride8) => C3
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        c3 = x

        # stage2 (stride16) => C4
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)
        c4 = x

        # stage3 (stride32) => C5
        x = self.downsample_layers[3](x)
        x = self.stages[3](x)
        c5 = x

        return {"c3": c3, "c4": c4, "c5": c5}