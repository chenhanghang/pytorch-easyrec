import torch
import torch.nn as nn
from models.base.layer import FeaturesEmbedding,MLP
from models.base.base_model import BaseModel

import torch
import torch.nn as nn


class CrossLayer(nn.Module):

    def __init__(self, width):
        super(CrossLayer, self).__init__()
        self.w = nn.Parameter(torch.zeros((width,)))
        self.bias = nn.Parameter(torch.zeros((1,)))
        nn.init.xavier_uniform_(self.w.unsqueeze(0).data)

    def forward(self, x0x1):
        x0, x1 = x0x1
        x2 = x0.mul(x1.mul(self.w).sum(dim=1, keepdim=True)) + x1 + self.bias
        return x0, x2


class CrossNetwork(nn.Module):

    def __init__(self, width, num_layers=3):
        super(CrossNetwork, self).__init__()
        self.layers = nn.Sequential(*[
            CrossLayer(width)
            for _ in range(num_layers)
        ])

    def forward(self, x0):
        # x size: (batch_size, num_fields)
        x0, output = self.layers((x0, x0))
        return output


class DeepCrossNetwork(BaseModel):

    def __init__(self, field_dims, embed_dim=4):
        super(DeepCrossNetwork, self).__init__()

        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.cross = CrossNetwork(len(field_dims) * embed_dim)
        self.deep = MLP([embed_dim * len(field_dims), 128, 64, 32])
        self.fc = nn.Linear(32 + embed_dim * len(field_dims), 1)

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        x = self.embedding(x).reshape(x.shape[0], -1)

        cross_output = self.cross(x)
        deep_output = self.deep(x)

        stacked = torch.hstack([cross_output, deep_output])
        output = self.fc(stacked)
        output = torch.sigmoid(output)

        return output

# Train Epoch: 5 [88704/90000 (99%)] Loss: 0.123598
#     epoch          : 5
#     loss           : 0.12028536635493352
#     auc            : 0.9874196018293684
#     val_loss       : 0.9516407741775995
#     val_auc        : 0.6857603116170549