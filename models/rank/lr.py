import torch
import torch.nn as nn
from models.base.layer import FeaturesEmbedding
from models.base.base_model import BaseModel

class LogisticRegression(BaseModel):
    
    def __init__(self, field_dims):
        super(LogisticRegression, self).__init__()
        
        self.bias = nn.Parameter(torch.zeros((1, )))
        self.embed = FeaturesEmbedding(field_dims, 1)

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        output = self.embed(x).sum(dim=1) + self.bias
        output = torch.sigmoid(output)
        return output

# epoch 5
# loss           : 0.3637032979167998
# auc            : 0.8843321827004119
# val_loss       : 0.45053261669376227
# val_auc        : 0.7647734188122769