# ===================== 2_models.py 完整内容 =====================
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- MLP --------------------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# -------------------- ResNet --------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.layers(x)
        out += residual
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(128, dropout) for _ in range(3)
        ])
        self.output_layer = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_layer(x)

# -------------------- FT-Transformer --------------------
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_model=64):
        super().__init__()
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(n_features)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, x):
        # x: [batch_size, n_features]
        batch_size = x.shape[0]
        # 每个特征单独嵌入: [batch_size, n_features, d_model]
        features = []
        for i, embedding in enumerate(self.feature_embeddings):
            features.append(embedding(x[:, i:i+1]))
        x = torch.stack(features, dim=1)
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, d_ffn=256, dropout=0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class FTTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, n_heads=4, d_ffn=256, dropout=0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(input_dim, d_model)
        self.transformers = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, d_ffn, dropout) for _ in range(3)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.tokenizer(x)
        x = self.transformers(x)
        x = self.norm(x)
        # 使用CLS token输出
        cls_output = x[:, 0]
        return self.head(cls_output)

