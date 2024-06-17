import torch
import torch.nn as nn
from models.base.layer import FeaturesEmbedding,EmbeddingsInteraction,MLP
from models.base.base_model import BaseModel

import torch
import torch.nn as nn

class InnerProduct(nn.Module):

    def __init__(self):
        super(InnerProduct, self).__init__()
        self.interaction = EmbeddingsInteraction()

    def forward(self, x):
        p = self.interaction(x).sum(dim=2)
        return p


class OuterProduct(nn.Module):

    def __init__(self):
        super(OuterProduct, self).__init__()

    def forward(self, x):
        num_fields = x.shape[1]
        field_dims = x.shape[2]

        sum_f = x.sum(dim=1)

        i1, i2 = [], []
        for i in range(field_dims):
            for j in range(field_dims):
                i1.append(i)
                i2.append(j)
        p = torch.mul(sum_f[:, i1], sum_f[:, i2])

        return p


class PNN(nn.Module):

    def __init__(self, field_dims, embed_dim=4, hidden_size=256, method='inner'):
        super(PNN, self).__init__()

        num_fields = len(field_dims)

        self.embed = FeaturesEmbedding(field_dims, embed_dim)

        if method == 'inner':
            self.pn = InnerProduct()
            mlp_input_size = num_fields * embed_dim + num_fields * (num_fields - 1) // 2
        elif method == 'outer':
            self.pn = OuterProduct()
            mlp_input_size = num_fields * embed_dim + embed_dim ** 2

        self.bias = nn.Parameter(torch.zeros((num_fields * embed_dim,)))
        nn.init.xavier_uniform_(self.bias.unsqueeze(0).data)

        self.mlp = MLP([mlp_input_size, hidden_size, 1])

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        # embed(x) shape: (batch_size, num_fields, embed_dim)

        x = self.embed(x)
        z = x.reshape(x.shape[0], -1)
        p = self.pn(x)

        output = torch.cat([z + self.bias, p], dim=1)
        output = self.mlp(output)
        output = torch.sigmoid(output)

        return output


# rain Epoch: 5 [87296/90000 (97%)] Loss: 0.153482
# Train Epoch: 5 [88704/90000 (99%)] Loss: 0.090566
#     epoch          : 5
#     loss           : 0.09421669745246287
#     auc            : 0.9923257354140769
#     val_loss       : 0.9727737692338002
#     val_auc        : 0.6896959159718495