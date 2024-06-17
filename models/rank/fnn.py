import torch
import torch.nn as nn
from models.base.layer import FeaturesEmbedding,MLP
from models.base.base_model import BaseModel

import torch
import torch.nn as nn

# 引入残差
class FNN(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(FNN, self).__init__()

        # w1, w2, ..., wn
        self.embed1 = FeaturesEmbedding(field_dims, 1)

        # v1, v2, ..., vn
        self.embed2 = FeaturesEmbedding(field_dims, embed_dim)

        self.mlp = MLP([(embed_dim + 1) * len(field_dims), 128, 64, 32, 1])

    def forward(self, x):
        # x shape: (batch_size, num_fields)

        w = self.embed1(x).squeeze(-1)
        v = self.embed2(x).reshape(x.shape[0], -1)
        stacked = torch.hstack([w, v])

        output = self.mlp(stacked)
        output = torch.sigmoid(output)
        return output

# 还是深度牛 
#Train Epoch: 5 [88704/90000 (99%)] Loss: 0.097049
#     epoch          : 5
#     loss           : 0.12688309071712534
#     auc            : 0.9859080953760817
#     val_loss       : 0.8323919690107997
#     val_auc        : 0.7034457141754007