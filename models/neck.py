# Chứa AIFI và CCFF
# models/neck.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding2D(nn.Module):
    def __init__(self, dim=256, temperature=10000):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, x):
        # x: (N,C,H,W)
        n, c, h, w = x.shape
        device = x.device
        y_embed = torch.linspace(0, 1, h, device=device).unsqueeze(1).repeat(1, w)
        x_embed = torch.linspace(0, 1, w, device=device).unsqueeze(0).repeat(h, 1)

        dim_t = torch.arange(self.dim // 2, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.dim // 2))

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)

        pos = torch.cat((pos_y, pos_x), dim=-1)  # (H,W,dim)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(n, 1, 1, 1)  # (N,dim,H,W)
        return pos

class HybridEncoder(nn.Module):
    """
    Input: dict(c3,c4,c5) -> Output: dict(p3,p4,p5) đã encode.
    """
    def __init__(self, in_channels, hidden_dim=256, nhead=8, num_layers=2, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.proj3 = nn.Conv2d(in_channels["c3"], hidden_dim, 1)
        self.proj4 = nn.Conv2d(in_channels["c4"], hidden_dim, 1)
        self.proj5 = nn.Conv2d(in_channels["c5"], hidden_dim, 1)

        # FPN-like top-down
        self.lateral4 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.lateral3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding2D(dim=hidden_dim)

    def _flatten(self, feat, pos):
        # feat: (N,C,H,W) -> tokens: (N, HW, C)
        n, c, h, w = feat.shape
        feat_t = feat.flatten(2).transpose(1, 2)
        pos_t = pos.flatten(2).transpose(1, 2)
        return feat_t, pos_t, (h, w)

    def _unflatten(self, tokens, hw):
        # tokens: (N, HW, C) -> (N,C,H,W)
        h, w = hw
        return tokens.transpose(1, 2).reshape(tokens.size(0), -1, h, w)

    def forward(self, feats):
        c3, c4, c5 = feats["c3"], feats["c4"], feats["c5"]
        p3 = self.proj3(c3)
        p4 = self.proj4(c4)
        p5 = self.proj5(c5)

        # top-down fusion (nhẹ)
        p4 = self.lateral4(p4 + F.interpolate(p5, size=p4.shape[-2:], mode="nearest"))
        p3 = self.lateral3(p3 + F.interpolate(p4, size=p3.shape[-2:], mode="nearest"))

        # transformer encoder trên concat tokens multi-scale
        pos3 = self.pos(p3); pos4 = self.pos(p4); pos5 = self.pos(p5)
        t3, ps3, hw3 = self._flatten(p3, pos3)
        t4, ps4, hw4 = self._flatten(p4, pos4)
        t5, ps5, hw5 = self._flatten(p5, pos5)

        tokens = torch.cat([t3 + ps3, t4 + ps4, t5 + ps5], dim=1)  # (N, sumHW, C)
        enc = self.encoder(tokens)

        # tách lại
        n3 = t3.size(1); n4 = t4.size(1); n5 = t5.size(1)
        e3 = enc[:, :n3]; e4 = enc[:, n3:n3+n4]; e5 = enc[:, n3+n4:]
        p3 = self._unflatten(e3, hw3)
        p4 = self._unflatten(e4, hw4)
        p5 = self._unflatten(e5, hw5)

        return {"p3": p3, "p4": p4, "p5": p5}