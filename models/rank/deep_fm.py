import torch
import torch.nn as nn
from models.base.layer import FeaturesEmbedding,MLP,EmbeddingsInteraction
from models.base.base_model import BaseModel

# FM 基础上引入深度DNN
class DeepFM(BaseModel):
    
    def __init__(self, field_dims, embed_dim=4):
        super(DeepFM, self).__init__()
        
        num_fileds = len(field_dims)
        
        self.embed1 = FeaturesEmbedding(field_dims, 1)
        self.embed2 = FeaturesEmbedding(field_dims, embed_dim)
        
        self.fm = EmbeddingsInteraction()
        
        self.deep = MLP([embed_dim * num_fileds, 128, 64, 32])
        self.fc = nn.Linear(1 + num_fileds * (num_fileds - 1) // 2 + 32, 1)

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        # embed(x) shape: (batch_size, num_fields, embed_dim)

        embeddings = self.embed2(x)
        embeddings_cross = self.fm(embeddings).sum(dim=-1)
        deep_output = self.deep(embeddings.reshape(x.shape[0], -1))
        
        stacked = torch.hstack([self.embed1(x).sum(dim=1), embeddings_cross, deep_output])
        output = self.fc(stacked)
        output = torch.sigmoid(output)
        return output

   # epoch          : 5
   #  loss           : 0.12242570373928174
   #  auc            : 0.9868901232011532
   #  val_loss       : 0.9295702056039737
   #  val_auc        : 0.6976871678842516