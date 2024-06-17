import torch
import torch.nn as nn
from models.base.layer import FeaturesEmbedding
from models.base.base_model import BaseModel

import torch
import torch.nn as nn

# 引入二阶特征交叉
class FM(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(FM, self).__init__()

        self.embed1 = FeaturesEmbedding(field_dims, 1)
        self.embed2 = FeaturesEmbedding(field_dims, embed_dim)
        self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        # embed(x) shape: (batch_size, num_fields, embed_dim)
        square_sum = self.embed2(x).sum(dim=1).pow(2).sum(dim=1)
        sum_square = self.embed2(x).pow(2).sum(dim=1).sum(dim=1)
        output = self.embed1(x).squeeze(-1).sum(dim=1) + self.bias + (square_sum - sum_square) / 2
        output = torch.sigmoid(output).unsqueeze(-1)
        return output

# Train Epoch: 5 [87296/90000 (97%)] Loss: 0.224228
# Train Epoch: 5 [88704/90000 (99%)] Loss: 0.309949
#     epoch          : 5
#     loss           : 0.21405346983705054
#     auc            : 0.9566970821378984
#     val_loss       : 0.5802808468100391
#     val_auc        : 0.716359182961319