# Chứa ConvNeXt (timm)
import torch.nn as nn
import timm

class ConvNeXtBackbone(nn.Module):
    def __init__(self, model_name='convnext_tiny.fb_in1k'):
        super().__init__()
        # features_only=True giúp lấy đặc trưng ở các stage khác nhau
        self.model = timm.create_model(model_name, pretrained=True, features_only=True)
        # Kênh đầu ra mặc định của ConvNeXt-T thường là [96, 192, 384, 768]
        self.out_channels = [192, 384, 768] # Lấy stage 2, 3, 4

    def forward(self, x):
        feats = self.model(x)
        return feats[1:] # Trả về list các tensor đặc trưng đa quy mô