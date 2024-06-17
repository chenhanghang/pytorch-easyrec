import torch
import torch.nn as nn
from models.base.layer import FeaturesEmbedding,MLP
from models.base.base_model import BaseModel

# class WideDeep(nn.Module):
    
#     def __init__(self, field_dims, embed_dim=4):
#         super(WideDeep, self).__init__()
        
#         self.wide = FeaturesEmbedding(field_dims, 1)
        
#         self.embedding = FeaturesEmbedding(field_dims, embed_dim)
#         self.deep = MLP([embed_dim * len(field_dims), 128, 64, 32])
#         #self.fc = nn.Linear(32 + embed_dim * len(field_dims), 1)
#         self.fc = nn.Linear(32 + len(field_dims), 1)
        
#     def forward(self, x):
#         # x shape: (batch_size, num_fields)
#         wide_output = self.wide(x)
#         embedding_output = self.embedding(x).reshape(x.shape[0], -1)
#         deep_output = self.deep(embedding_output)
#         concat = torch.hstack([embedding_output, deep_output])
#         output = self.fc(concat)
#         output = torch.sigmoid(output)
        
#         return output

    # epoch          : 5
    # loss           : 0.1208573346640068
    # auc            : 0.987044586077987
    # val_loss       : 0.9309854164153715
    # val_auc        : 0.6993226113144766

# 记忆与泛化的结合
class WideDeep(BaseModel):
    
    def __init__(self, field_dims, embed_dim=4):
        super(WideDeep, self).__init__()
        
        self.wide = FeaturesEmbedding(field_dims, 1)
        
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.deep = MLP([embed_dim * len(field_dims), 128, 64, 32])
        #self.fc = nn.Linear(32 + embed_dim * len(field_dims), 1)
        self.fc = nn.Linear(32 + len(field_dims), 1)
        
    def forward(self, x):
        # x shape: (batch_size, num_fields)
        wide_output = self.wide(x) #[128, 39, 1]
        wide_output = wide_output.squeeze()
        embedding_output = self.embedding(x).reshape(x.shape[0], -1)
        deep_output = self.deep(embedding_output)
        concat = torch.cat([wide_output, deep_output],1)
        output = self.fc(concat)
        output = torch.sigmoid(output)
        
        return output

# epoch          : 5
#     loss           : 0.12004293506668712
#     auc            : 0.9872757777941541
#     val_loss       : 0.8850089198426355
#     val_auc        : 0.697772662552376