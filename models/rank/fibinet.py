import torch
import torch.nn as nn
from models.base.layer import FeaturesEmbedding,MLP,SENETLayer,BiLinearInteractionLayer
from models.base.base_model import BaseModel


class FiBiNet(BaseModel):
    """
        Args:
        features (list[Feature Class]): training by the whole module.
        reduction_ratio (int) : Hidden layer reduction factor of SENET layer
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
        bilinear_type (str): the type bilinear interaction function, in ["field_all", "field_each", "field_interaction"], field_all means that all features share a W, field_each means that a feature field corresponds to a W_i, field_interaction means that a feature field intersection corresponds to a W_ij
    """
    def __init__(self, field_dims, embed_dim=4, reduction_ratio=3, bilinear_type="field_all", **kwargs):
        super(FiBiNet, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        num_fields = len(field_dims)
        self.senet_layer = SENETLayer(num_fields, reduction_ratio)
        self.bilinear_interaction = BiLinearInteractionLayer(embed_dim, num_fields,  bilinear_type)
        self.dims = num_fields * (num_fields - 1) * embed_dim
        self.mlp = MLP([self.dims, 128, 64, 32, 1])

    def forward(self, x):
        embed_x = self.embedding(x)
        embed_senet = self.senet_layer(embed_x)
        embed_bi1 = self.bilinear_interaction(embed_x)
        embed_bi2 = self.bilinear_interaction(embed_senet)
        shallow_part = torch.flatten(torch.cat([embed_bi1, embed_bi2], dim=1), start_dim=1)
        mlp_out = self.mlp(shallow_part)
        return torch.sigmoid(mlp_out)

# Train Epoch: 5 [87296/90000 (97%)] Loss: 0.155484
# Train Epoch: 5 [88704/90000 (99%)] Loss: 0.078413
#     epoch          : 5
#     loss           : 0.09814038953944956
#     auc            : 0.991757191075608
#     val_loss       : 0.8467474972145467
#     val_auc        : 0.7086793095112859

# field_all模式
# Train Epoch: 5 [88704/90000 (99%)] Loss: 0.075375
#     epoch          : 5
#     loss           : 0.10736974304414947
#     auc            : 0.9900422205097954
#     val_loss       : 0.8425085355963888
#     val_auc        : 0.6978360704818809