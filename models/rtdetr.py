# Class tổng hợp mô hình
# models/rtdetr.py
# models/rtdetr.py
import torch.nn as nn
from .backbone import ConvNeXtTiny  # Sửa lại đúng tên class
from .neck import HybridEncoder
from .head import RTDETRDecoderHead

class RTDETR(nn.Module):
    def __init__(self, num_classes=80, hidden_dim=256, num_queries=300):
        super().__init__()
        self.backbone = ConvNeXtBackbone() # Sửa ở đây
        self.neck = HybridEncoder(self.backbone.out_channels, hidden_dim=hidden_dim)
        self.head = RTDETRDecoderHead(hidden_dim=hidden_dim, num_classes=num_classes, num_queries=num_queries)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        out = self.head(feats)
        return out