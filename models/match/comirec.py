"""
Date: create on 07/06/2022
References:
    paper: Controllable Multi-Interest Framework for Recommendation
    url: https://arxiv.org/pdf/2005.09347.pdf
    code: https://github.com/ShiningCosmos/pytorch_ComiRec/blob/main/ComiRec.py
Authors: Kai Wang, 306178200@qq.com
"""

import torch

from torch import nn
import torch.nn.functional as F
from models.base.layer import MLP_MUL,EmbeddingLayer


class ComirecSA(torch.nn.Module):
    """The match model mentioned in `Controllable Multi-Interest Framework for Recommendation` paper.
    It's a ComirecSA match model trained by global softmax loss on list-wise samples.
    Note in origin paper, it's without item dnn tower and train item embedding directly.

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        history_features (list[Feature Class]): training history
        item_features (list[Feature Class]): training by the embedding table, it's the item id feature.
        neg_item_feature (list[Feature Class]): training by the embedding table, it's the negative items id feature.
        temperature (float): temperature factor for similarity score, default to 1.0.
        interest_num （int): interest num
    """

    def __init__(self, user_features, history_features, item_features, neg_item_feature, temperature=1.0, interest_num=4):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.history_features = history_features
        self.neg_item_feature = neg_item_feature
        self.temperature = temperature
        self.interest_num = interest_num
        self.user_dims = sum([fea.embed_dim for fea in user_features+history_features])

        self.embedding = EmbeddingLayer(user_features + item_features + history_features)
        self.multi_interest_sa = MultiInterestSA(embedding_dim=self.history_features[0].embed_dim, interest_num=self.interest_num)
        self.convert_user_weight = nn.Parameter(torch.rand(self.user_dims, self.history_features[0].embed_dim), requires_grad=True)
        self.mode = None

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        pos_item_embedding = item_embedding[:, 0, :]  # b*emb
        # b* interest_num * emb b*emb*1
        dot_res = torch.bmm(user_embedding, pos_item_embedding.squeeze(1).unsqueeze(-1))
        k_index = torch.argmax(dot_res, dim=1)
        best_interest_emb = torch.rand(user_embedding.shape[0], user_embedding.shape[2]).to(user_embedding.device)
        for k in range(user_embedding.shape[0]):
            best_interest_emb[k, :] = user_embedding[k, k_index[k], :]
        best_interest_emb = best_interest_emb.unsqueeze(1)  # b*emb -> b*1*emb

        y = torch.mul(best_interest_emb, item_embedding).sum(dim=1)  # b*1*emb mul b*k*emb -> b*k*emb -> b*k  broadcast 机制

        return y

    def user_tower(self, x):
        if self.mode == "item":
            return None
        input_user = self.embedding(x, self.user_features, squeeze_dim=True).unsqueeze(1)  # [batch_size, num_features*deep_dims]
        input_user = input_user.expand([input_user.shape[0], self.interest_num, input_user.shape[-1]]) # batch_size, K, dims

        history_emb = self.embedding(x, self.history_features).squeeze(1) # batch_size, s, emb
        mask = self.gen_mask(x)
        mask = mask.unsqueeze(-1).float() # b*1*his_len
        multi_interest_emb = self.multi_interest_sa(history_emb,mask)

        input_user = torch.cat([input_user,multi_interest_emb],dim=-1)

        # user_embedding = self.user_mlp(input_user).unsqueeze(1)  #[batch_size, interest_num, embed_dim]
        user_embedding = torch.matmul(input_user,self.convert_user_weight) # batch_size*embed_dim
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)  # L2 normalize
        if self.mode == "user":
            return user_embedding  #inference embedding mode -> [batch_size, interest_num, embed_dim]
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        pos_embedding = self.embedding(x, self.item_features, squeeze_dim=False)  #[batch_size, 1, embed_dim]，此处只有一个特征
        pos_embedding = F.normalize(pos_embedding, p=2, dim=-1)  # L2 normalize
        if self.mode == "item":  #inference embedding mode
            return pos_embedding.squeeze(1)  #[batch_size, embed_dim]
        neg_embeddings = self.embedding(x, self.neg_item_feature,
                                        squeeze_dim=False).squeeze(1)  #[batch_size, 1，n_neg_items, embed_dim] -》[batch_size, n_neg_items, embed_dim]
        neg_embeddings = F.normalize(neg_embeddings, p=2, dim=-1)  # L2 normalize
        return torch.cat((pos_embedding, neg_embeddings), dim=1)  #[batch_size, 1+n_neg_items, embed_dim]

    def gen_mask(self, x):
        his_list = x[self.history_features[0].name]
        mask = (his_list > 0).long() # b*his_len
        return mask

class ComirecDR(torch.nn.Module):
    """The match model mentioned in `Controllable Multi-Interest Framework for Recommendation` paper.
    It's a ComirecDR match model trained by global softmax loss on list-wise samples.
    Note in origin paper, it's without item dnn tower and train item embedding directly.

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        history_features (list[Feature Class]): training history
        item_features (list[Feature Class]): training by the embedding table, it's the item id feature.
        neg_item_feature (list[Feature Class]): training by the embedding table, it's the negative items id feature.
        max_length (int): max sequence length of input item sequence
        temperature (float): temperature factor for similarity score, default to 1.0.
        interest_num （int): interest num
    """

    def __init__(self, user_features, history_features, item_features, neg_item_feature, max_length, temperature=1.0, interest_num=4):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.history_features = history_features
        self.neg_item_feature = neg_item_feature
        self.temperature = temperature
        self.interest_num = interest_num
        self.max_length = max_length
        self.user_dims = sum([fea.embed_dim for fea in user_features+history_features])

        self.embedding = EmbeddingLayer(user_features + item_features + history_features)
        self.capsule = CapsuleNetwork(self.history_features[0].embed_dim,self.max_length,bilinear_type=2,interest_num=self.interest_num)
        self.convert_user_weight = nn.Parameter(torch.rand(self.user_dims, self.history_features[0].embed_dim), requires_grad=True)
        self.mode = None

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        pos_item_embedding = item_embedding[:,0,:]
        dot_res = torch.bmm(user_embedding, pos_item_embedding.squeeze(1).unsqueeze(-1))
        k_index = torch.argmax(dot_res, dim=1)
        best_interest_emb = torch.rand(user_embedding.shape[0], user_embedding.shape[2]).to(user_embedding.device)
        for k in range(user_embedding.shape[0]):
            best_interest_emb[k, :] = user_embedding[k, k_index[k], :]
        best_interest_emb = best_interest_emb.unsqueeze(1)

        y = torch.mul(best_interest_emb, item_embedding).sum(dim=1)

        return y

    def user_tower(self, x):
        if self.mode == "item":
            return None
        input_user = self.embedding(x, self.user_features, squeeze_dim=True).unsqueeze(1)  #[batch_size, num_features*deep_dims]
        input_user = input_user.expand([input_user.shape[0], self.interest_num, input_user.shape[-1]])

        history_emb = self.embedding(x, self.history_features).squeeze(1)
        mask = self.gen_mask(x)
        multi_interest_emb = self.capsule(history_emb,mask)

        input_user = torch.cat([input_user,multi_interest_emb],dim=-1)

        # user_embedding = self.user_mlp(input_user).unsqueeze(1)  #[batch_size, interest_num, embed_dim]
        user_embedding = torch.matmul(input_user,self.convert_user_weight)
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)  # L2 normalize
        if self.mode == "user":
            return user_embedding  #inference embedding mode -> [batch_size, interest_num, embed_dim]
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        pos_embedding = self.embedding(x, self.item_features, squeeze_dim=False)  #[batch_size, 1, embed_dim]
        pos_embedding = F.normalize(pos_embedding, p=2, dim=-1)  # L2 normalize
        if self.mode == "item":  #inference embedding mode
            return pos_embedding.squeeze(1)  #[batch_size, embed_dim]
        neg_embeddings = self.embedding(x, self.neg_item_feature,
                                        squeeze_dim=False).squeeze(1)  #[batch_size, n_neg_items, embed_dim]
        neg_embeddings = F.normalize(neg_embeddings, p=2, dim=-1)  # L2 normalize
        return torch.cat((pos_embedding, neg_embeddings), dim=1)  #[batch_size, 1+n_neg_items, embed_dim]

    def gen_mask(self, x):
        his_list = x[self.history_features[0].name]
        mask = (his_list > 0).long()
        return mask


####################################

class MultiInterestSA(nn.Module):
    """MultiInterest Attention mentioned in the Comirec paper.

    Args:
        embedding_dim (int): embedding dim of item embedding
        interest_num (int): num of interest
        hidden_dim (int): hidden dim

    Shape:
        - Input: seq_emb : (batch,seq,emb)
                 mask : (batch,seq,1)
        - Output: `(batch_size, interest_num, embedding_dim)`

    """

    def __init__(self, embedding_dim, interest_num, hidden_dim=None):
        super(MultiInterestSA, self).__init__()
        self.embedding_dim = embedding_dim
        self.interest_num = interest_num
        if hidden_dim == None:
            self.hidden_dim = self.embedding_dim * 4
        self.W1 = torch.nn.Parameter(torch.rand(self.embedding_dim, self.hidden_dim), requires_grad=True)
        self.W2 = torch.nn.Parameter(torch.rand(self.hidden_dim, self.interest_num), requires_grad=True)
        self.W3 = torch.nn.Parameter(torch.rand(self.embedding_dim, self.embedding_dim), requires_grad=True)

    def forward(self, seq_emb, mask=None):
        H = torch.einsum('bse, ed -> bsd', seq_emb, self.W1).tanh() # seq_emb:[b,s,e] -> [b,s,hid]
        if mask != None:
            A = torch.einsum('bsd, dk -> bsk', H, self.W2) + -1.e9 * (1 - mask.float()) # b,s,interest_num
            A = F.softmax(A, dim=1)
        else:
            A = F.softmax(torch.einsum('bsd, dk -> bsk', H, self.W2), dim=1)
        A = A.permute(0, 2, 1) # b,interest_num,s
        multi_interest_emb = torch.matmul(A, seq_emb)  # [b,interest_num,s]* [b,s,e] -> [b,interest_num,emb]
        return multi_interest_emb

class CapsuleNetwork(nn.Module):
    """CapsuleNetwork mentioned in the Comirec and MIND paper.

    Args:
        hidden_size (int): embedding dim of item embedding
        seq_len (int): length of the item sequence
        bilinear_type (int): 0 for MIND, 2 for ComirecDR
        interest_num (int): num of interest
        routing_times (int): routing times

    Shape:
        - Input: seq_emb : (batch,seq,emb)
                 mask : (batch,seq,1)
        - Output: `(batch_size, interest_num, embedding_dim)`

    """

    def __init__(self, embedding_dim, seq_len, bilinear_type=2, interest_num=4, routing_times=3, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.embedding_dim = embedding_dim  # h
        self.seq_len = seq_len  # s
        self.bilinear_type = bilinear_type
        self.interest_num = interest_num
        self.routing_times = routing_times

        self.relu_layer = relu_layer
        self.stop_grad = True
        self.relu = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), nn.ReLU())
        if self.bilinear_type == 0:  # MIND
            self.linear = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        elif self.bilinear_type == 1:
            self.linear = nn.Linear(self.embedding_dim, self.embedding_dim * self.interest_num, bias=False)
        else:
            self.w = nn.Parameter(torch.Tensor(1, self.seq_len, self.interest_num * self.embedding_dim, self.embedding_dim))

    def forward(self, item_eb, mask):
        if self.bilinear_type == 0:
            item_eb_hat = self.linear(item_eb)
            item_eb_hat = item_eb_hat.repeat(1, 1, self.interest_num)
        elif self.bilinear_type == 1:
            item_eb_hat = self.linear(item_eb)
        else:
            u = torch.unsqueeze(item_eb, dim=2)
            item_eb_hat = torch.sum(self.w[:, :self.seq_len, :, :] * u, dim=3)

        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.seq_len, self.interest_num, self.embedding_dim))
        item_eb_hat = torch.transpose(item_eb_hat, 1, 2).contiguous()
        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.interest_num, self.seq_len, self.embedding_dim))

        if self.stop_grad:
            item_eb_hat_iter = item_eb_hat.detach()
        else:
            item_eb_hat_iter = item_eb_hat

        if self.bilinear_type > 0:
            capsule_weight = torch.zeros(item_eb_hat.shape[0],
                                         self.interest_num,
                                         self.seq_len,
                                         device=item_eb.device,
                                         requires_grad=False)
        else:
            capsule_weight = torch.randn(item_eb_hat.shape[0],
                                         self.interest_num,
                                         self.seq_len,
                                         device=item_eb.device,
                                         requires_grad=False)

        for i in range(self.routing_times):  # 动态路由传播3次
            atten_mask = torch.unsqueeze(mask, 1).repeat(1, self.interest_num, 1)
            paddings = torch.zeros_like(atten_mask, dtype=torch.float)

            capsule_softmax_weight = F.softmax(capsule_weight, dim=-1)
            capsule_softmax_weight = torch.where(torch.eq(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)
            if i < 2:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat_iter)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = torch.matmul(item_eb_hat_iter, torch.transpose(interest_capsule, 2, 3).contiguous())
                delta_weight = torch.reshape(delta_weight, (-1, self.interest_num, self.seq_len))
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

            interest_capsule = torch.reshape(interest_capsule, (-1, self.interest_num, self.embedding_dim))

            if self.relu_layer:
                interest_capsule = self.relu(interest_capsule)

            return interest_capsule
