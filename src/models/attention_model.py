import torch.nn as nn
import torch.nn.functional as F

class InputEmbeddingPosEncoding(nn.Module):
    def __init__(self, feature_dim=512):
        super(InputEmbeddingPosEncoding, self).__init__()
        self.lin_proj_layer = nn.Linear(in_features=156, out_features=1500)
        self.lin_proj_layer1 = nn.Linear(in_features=1500, out_features=feature_dim)
        # self.lin_proj_layer2 = nn.Linear(in_features=500, out_features=256)
        self.pos_encoder = AbsolutePositionalEncoding()

    def forward(self, x):
        x = self.lin_proj_layer(x)
        x = self.lin_proj_layer1(x)
        # x = self.lin_proj_layer2(x)
        x = self.pos_encoder(x)
        return x

class AbsolutePositionalEncoding(nn.Module):
    def __init__(self):
        super(AbsolutePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, feature_dim=512):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(feature_dim, num_heads=10)
        self.linear1 = nn.Linear(feature_dim, 2048)
        self.dropout1 = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(2048, feature_dim)
        self.dropout2 = nn.Dropout(p=0.1)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x, _ = self.self_attn(x, x, x)
        x = x.permute(1, 0, 2)
        residual = x
        x = self.norm1(x)
        x = F.relu(self.linear1(x))
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x += residual
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, feature_dim=512):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(feature_dim) for _ in range(4)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransformerEncoderNetwork(nn.Module):
    def __init__(self, feature_dim=512):
        super(TransformerEncoderNetwork, self).__init__()
        self.emb = InputEmbeddingPosEncoding(feature_dim)
        self.transformer_encoder = TransformerEncoder(feature_dim)
        
    def forward(self, x):
        x = self.emb(x)
        x = self.transformer_encoder(x)
        
        return x