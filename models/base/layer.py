import numpy as np
import torch
import torch.nn as nn
from itertools import combinations
from models.base.features import DenseFeature, SparseFeature, SequenceFeature
from models.base.activation import activation_layer


# 在 cpu 下，比 nn.Embedding 快，但是在 gpu 的序列模型下比后者慢太多了
class CpuEmbedding(nn.Module):
    def __init__(self, num_embeddings, embed_dim):
        super(CpuEmbedding, self).__init__()

        self.weight = nn.Parameter(torch.zeros((num_embeddings, embed_dim)))
        nn.init.xavier_uniform_(self.weight.data)

    def forward(self, x):
        """
        :param x: shape (batch_size, num_fields)
        :return: shape (batch_size, num_fields, embedding_dim)
        """
        return self.weight[x]


class Embedding:
    def __new__(cls, num_embeddings, embed_dim):
        if torch.cuda.is_available():
            embedding = nn.Embedding(num_embeddings, embed_dim)
            nn.init.xavier_uniform_(embedding.weight.data)
            return embedding
        else:
            return CpuEmbedding(num_embeddings, embed_dim)

# 对 离散特征embdding
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims, embed_dim):
        super(FeaturesEmbedding, self).__init__()
        self.embedding = Embedding(sum(field_dims), embed_dim)

        # e.g. field_dims = [2, 3, 4, 5], offsets = [0, 2, 5, 9]
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: shape (batch_size, num_fields)
        :return: shape (batch_size, num_fields, embedding_dim)
        """
        x = x + x.new_tensor(self.offsets)
        return self.embedding(x)

#深度网络
class MLP(nn.Module):
    def __init__(self, layer, batch_norm=True):
        super(MLP, self).__init__()
        layers = []
        input_size = layer[0]
        for output_size in layer[1: -1]:
            layers.append(nn.Linear(input_size, output_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(output_size))
            layers.append(nn.ReLU())
            input_size = output_size
        layers.append(nn.Linear(input_size, layer[-1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ConcatPooling(nn.Module):
    """Keep the origin sequence embedding shape

    Shape:
    - Input: `(batch_size, seq_length, embed_dim)`
    - Output: `(batch_size, seq_length, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x


class AveragePooling(nn.Module):
    """Pooling the sequence embedding matrix by `mean`.

    Shape:
        - Input
            x: `(batch_size, seq_length, embed_dim)`
            mask: `(batch_size, 1, seq_length)`
        - Output: `(batch_size, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask == None:
            return torch.mean(x, dim=1)
        else:
            sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)
            non_padding_length = mask.sum(dim=-1)
            return sum_pooling_matrix / (non_padding_length.float() + 1e-16)


class SumPooling(nn.Module):
    """Pooling the sequence embedding matrix by `sum`.

    Shape:
        - Input
            x: `(batch_size, seq_length, embed_dim)`
            mask: `(batch_size, 1, seq_length)`
        - Output: `(batch_size, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask == None:
            return torch.sum(x, dim=1)
        else:
            return torch.bmm(mask, x).squeeze(1)

class InputMask(nn.Module):
    """Return inputs mask from given features

    Shape:
        - Input: 
            x (dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape:`(batch_size, seq_len)`,\
                      sparse/dense feature value is a 1D tensor with shape `(batch_size)`.
            features (list or SparseFeature or SequenceFeature): Note that the elements in features are either all instances of SparseFeature or all instances of SequenceFeature.
        - Output: 
            - if input Sparse: `(batch_size, num_features)`
            - if input Sequence: `(batch_size, num_features_seq, seq_length)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, features):
        mask = []
        if not isinstance(features, list):
            features = [features]
        for fea in features:
            if isinstance(fea, SparseFeature) or isinstance(fea, SequenceFeature):
                if fea.padding_idx != None:
                    fea_mask = x[fea.name].long() != fea.padding_idx
                else:
                    fea_mask = x[fea.name].long() != -1
                mask.append(fea_mask.unsqueeze(1).float())
            else:
                raise ValueError("Only SparseFeature or SequenceFeature support to get mask.")
        return torch.cat(mask, dim=1)


class PredictionLayer(nn.Module):
    """Prediction Layer.
    Args:
        task_type (str): if `task_type='classification'`, then return sigmoid(x), 
                    change the input logits to probability. if`task_type='regression'`, then return x.
    """

    def __init__(self, task_type='classification'):
        super(PredictionLayer, self).__init__()
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be classification or regression")
        self.task_type = task_type

    def forward(self, x):
        if self.task_type == "classification":
            x = torch.sigmoid(x)
        return x
        
class MLP_MUL(nn.Module):
    """Multi Layer Perceptron Module, it is the most widely used module for 
    learning feature. Note we default add `BatchNorm1d` and `Activation` 
    `Dropout` for each `Linear` Module.

    Args:
        input dim (int): input size of the first Linear Layer.
        output_layer (bool): whether this MLP module is the output layer. If `True`, then append one Linear(*,1) module. 
        dims (list): output size of Linear Layer (default=[]).
        dropout (float): probability of an element to be zeroed (default = 0.5).
        activation (str): the activation function, support `[sigmoid, relu, prelu, dice, softmax]` (default='relu').

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)` or `(batch_size, dims[-1])`
    """

    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0, activation="relu"):
        super().__init__()
        if dims is None:
            dims = []
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(activation_layer(activation))
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class EmbeddingLayer(nn.Module):
    """General Embedding Layer.
    We save all the feature embeddings in embed_dict: `{feature_name : embedding table}`.
    Args:
        features (list): the list of `Feature Class`. It is means all the features which we want to create a embedding table.

    Shape:
        - Input: 
            x (dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape:`(batch_size, seq_len)`,\
                      sparse/dense feature value is a 1D tensor with shape `(batch_size)`.
            features (list): the list of `Feature Class`. It is means the current features which we want to do embedding lookup.
            squeeze_dim (bool): whether to squeeze dim of output (default = `False`).
        - Output: 
            - if input Dense: `(batch_size, num_features_dense)`.
            - if input Sparse: `(batch_size, num_features, embed_dim)` or  `(batch_size, num_features * embed_dim)`.
            - if input Sequence: same with input sparse or `(batch_size, num_features_seq, seq_length, embed_dim)` when `pooling=="concat"`.
            - if input Dense and Sparse/Sequence: `(batch_size, num_features_sparse * embed_dim)`. Note we must squeeze_dim for concat dense value with sparse embedding.
    """
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.embed_dict = nn.ModuleDict()
        self.n_dense = 0

        for fea in features:
            if fea.name in self.embed_dict:  # exist
                continue
            if isinstance(fea, SparseFeature) and fea.shared_with == None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, SequenceFeature) and fea.shared_with == None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, DenseFeature):
                self.n_dense += 1
    
    def forward(self, x, features, squeeze_dim=False):
        sparse_emb, dense_values = [], []
        sparse_exists, dense_exists = False, False
        for fea in features:
            if isinstance(fea, SparseFeature):
                if fea.shared_with == None:
                    sparse_emb.append(self.embed_dict[fea.name](x[fea.name].long()).unsqueeze(1)) # b * emb -> b * 1 * emb
                else:
                    sparse_emb.append(self.embed_dict[fea.shared_with](x[fea.name].long()).unsqueeze(1))
            elif isinstance(fea, SequenceFeature):
                if fea.pooling == "sum":
                    pooling_layer = SumPooling()
                elif fea.pooling == "mean":
                    pooling_layer = AveragePooling()
                elif fea.pooling == "concat":
                    pooling_layer = ConcatPooling()
                else:
                    raise ValueError("Sequence pooling method supports only pooling in %s, got %s." %
                                     (["sum", "mean"], fea.pooling))
                fea_mask = InputMask()(x, fea)
                if fea.shared_with == None:
                    sparse_emb.append(pooling_layer(self.embed_dict[fea.name](x[fea.name].long()), fea_mask).unsqueeze(1))
                else:
                    sparse_emb.append(pooling_layer(self.embed_dict[fea.shared_with](x[fea.name].long()), fea_mask).unsqueeze(1))  #shared specific sparse feature embedding
            else:
                dense_values.append(x[fea.name].float().unsqueeze(1))  #.unsqueeze(1).unsqueeze(1)

        if len(dense_values) > 0:
            dense_exists = True
            dense_values = torch.cat(dense_values, dim=1)
        if len(sparse_emb) > 0:
            sparse_exists = True
            sparse_emb = torch.cat(sparse_emb, dim=1)  #[batch_size, num_features, embed_dim]

        if squeeze_dim:  #Note: if the emb_dim of sparse features is different, we must squeeze_dim
            if dense_exists and not sparse_exists:  #only input dense features
                return dense_values
            elif not dense_exists and sparse_exists:
                return sparse_emb.flatten(start_dim=1)  #squeeze dim to : [batch_size, num_features*embed_dim]
            elif dense_exists and sparse_exists:
                return torch.cat((sparse_emb.flatten(start_dim=1), dense_values), dim=1)  #concat dense value with sparse embedding
            else:
                raise ValueError("The input features can note be empty")
        else:
            if sparse_exists:
                return sparse_emb  #[batch_size, num_features, embed_dim]
            else:
                raise ValueError(
                    "If keep the original shape:[batch_size, num_features, embed_dim], expected %s in feature list, got %s" %
                    ("SparseFeatures", features))
#交叉网络
class EmbeddingsInteraction(nn.Module):

    def __init__(self):
        super(EmbeddingsInteraction, self).__init__()

    def forward(self, x):
        """
        :param x: shape (batch_size, num_fields, embedding_dim)
        :return: shape (batch_size, num_fields*(num_fields)//2, embedding_dim)
        """

        num_fields = x.shape[1]
        i1, i2 = [], []
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                i1.append(i)
                i2.append(j)
        interaction = torch.mul(x[:, i1], x[:, i2])

        return interaction


class SENETLayer(nn.Module):
    """
    A weighted feature gating system in the SENet paper
    Args:
        num_fields (int): number of feature fields

    Shape:
        - num_fields: `(batch_size, *)`
        - Output: `(batch_size, *)`
    """
    def __init__(self, num_fields, reduction_ratio=3):
        super(SENETLayer, self).__init__()
        reduced_size = max(1, int(num_fields/ reduction_ratio))
        self.mlp = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False),
                                 nn.ReLU(),
                                 nn.Linear(reduced_size, num_fields, bias=False),
                                 nn.ReLU())
    def forward(self, x):
        z = torch.mean(x, dim=-1, out=None) # b * f * e -> b * f
        a = self.mlp(z)
        v = x*a.unsqueeze(-1) # 对应元素相乘 b * f * e  &&  b * f * 1
        return v

class BiLinearInteractionLayer(nn.Module):
    """
    Bilinear feature interaction module, which is an improved model of the FFM model
     Args:
        num_fields (int): number of feature fields
        bilinear_type(str): the type bilinear interaction function
    Shape:
        - num_fields: `(batch_size, *)`
        - Output: `(batch_size, *)`
    """
    def __init__(self, input_dim, num_fields, bilinear_type = "field_interaction"):
        super(BiLinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == "field_all":
            self.bilinear_layer = nn.Linear(input_dim, input_dim, bias=False)
        elif self.bilinear_type == "field_each":
            self.bilinear_layer = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for i in range(num_fields)])
        elif self.bilinear_type == "field_interaction":
            self.bilinear_layer = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for i,j in combinations(range(num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, x):
        feature_emb = torch.split(x, 1, dim=1) # 分割张量
        if self.bilinear_type == "field_all": # combinations 返回一个包含所有长度为 r 的组合的可迭代对象
            bilinear_list = [self.bilinear_layer(v_i)*v_j for v_i, v_j in combinations(feature_emb, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb[i])*feature_emb[j] for i,j in combinations(range(len(feature_emb)), 2)]
        elif self.bilinear_type == "field_interaction":
            bilinear_list = [self.bilinear_layer[i](v[0])*v[1] for i,v in enumerate(combinations(feature_emb, 2))]
        return torch.cat(bilinear_list, dim=1)
