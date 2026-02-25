# Chứa Transformer Decoder
# models/head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim]*(num_layers-1) + [out_dim]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < num_layers-1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class RTDETRDecoderHead(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=80, num_queries=300,
                 nhead=8, num_decoder_layers=6, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, feats):
        # feats: dict(p3,p4,p5) N,C,H,W
        p3, p4, p5 = feats["p3"], feats["p4"], feats["p5"]

        def flatten(x):
            return x.flatten(2).transpose(1, 2)  # N,HW,C

        memory = torch.cat([flatten(p3), flatten(p4), flatten(p5)], dim=1)  # N,S,C

        bs = memory.size(0)
        query = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)  # N,Q,C

        hs = self.decoder(tgt=query, memory=memory)  # N,Q,C

        logits = self.class_embed(hs)               # N,Q,K+1
        boxes = self.bbox_embed(hs).sigmoid()       # N,Q,4 in [0,1]
        return {"pred_logits": logits, "pred_boxes": boxes}