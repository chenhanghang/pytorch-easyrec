import torch
import torch.nn as nn

from models.base.layer import MLP_MUL, EmbeddingLayer,PredictionLayer

class SharedBottom(nn.Module):
    """Shared Bottom multi task model.

    Args:
        features (list): the list of `Feature Class`, training by the bottom and tower module.
        task_types (list): types of tasks, only support `["classfication", "regression"]`.
        bottom_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float}, keep `{"output_layer":False}`.
        tower_params_list (list): the list of tower params dict, the keys same as bottom_params.
    """
    def __init__(self, features, task_types, bottom_params, tower_params_list):
        super().__init__()
        self.features = features
        self.task_types = task_types
        self.embedding = EmbeddingLayer(features)
        self.bottom_dims = sum([fea.embed_dim for fea in features])

        self.bottom_mlp = MLP_MUL(self.bottom_dims, **{**bottom_params, **{"output_layer": False}})
        self.towers = nn.ModuleList(
            # input_dim, output_layer=True, dims=None, dropout=0, activation="relu"
            MLP_MUL(bottom_params["dims"][-1], **tower_params_list[i]) for i in range(len(task_types)))
        self.predict_layers = nn.ModuleList(PredictionLayer(task_type) for task_type in task_types) # 添加sigmoid

    def forward(self, x):
        input_bottom = self.embedding(x, self.features, squeeze_dim=True)
        x = self.bottom_mlp(input_bottom)

        ys = []
        for tower, predict_layer in zip(self.towers, self.predict_layers):
            tower_out = tower(x)
            y = predict_layer(tower_out)  #regression->keep, binary classification->sigmoid
            ys.append(y)
        return torch.cat(ys, dim=1)
