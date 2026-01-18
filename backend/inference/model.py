import torch
from torch import nn
import torch.nn.functional as F
from .path import CLF_PATH, EMBEDDING_PATH



class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.4)

    def forward(self, lstm_out, mask):
        # lstm_out: (B, T, H)
        # mask: (B, T, 1)

        scores = self.attn(lstm_out).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(mask.squeeze(-1) == 0, -1e9)

        attn_weights = F.softmax(scores, dim=1)  # (B, T)
        attn_weights = self.dropout(attn_weights)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)

        return context, attn_weights
    


class Model(nn.Module):
    def __init__(self, lstm_hidden=128, lstm_layers=1, num_classes=3, embed_proj_size=128):
        super().__init__()
        embedding_matrix = torch.load(EMBEDDING_PATH, map_location='cpu')['embedding']
        self.num_classes = num_classes

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.detach().clone().float(), 
            padding_idx=0, 
            freeze=True
        )

        self.embed_proj = nn.Sequential(
            nn.Linear(embedding_matrix.size(1), embed_proj_size),
            nn.LayerNorm(embed_proj_size),
            nn.Dropout(0.3)
        )

        self.lstm = nn.LSTM(
            input_size=embed_proj_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)

        self.attention = AttentionPooling(lstm_hidden * 2)

        self.layernorm = nn.LayerNorm(lstm_hidden * 2 + 4)

        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + 4, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, text_ids, extra_feats):
        # Embedding
        x = self.embedding(text_ids)  # (B, T, E)

        x = self.embed_proj(x)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, 2H)
        lstm_out = self.dropout(lstm_out)

        # Mask padding
        mask = (text_ids != 0).unsqueeze(-1)  # (B, T, 1)

        # Attention
        context, attn_weights = self.attention(lstm_out, mask)

        # Concatenate extra features
        features = torch.cat([context, extra_feats], dim=1)

        features = self.layernorm(features)

        logits = self.mlp(features)

        return logits



    def create_model():
        LSTM_HIDDEN = 128
        LSTM_LAYERS = 1
        EMBED_PROJ_SIZE = 128

        DEVIVE = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = Model(lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, embed_proj_size=EMBED_PROJ_SIZE).to(DEVIVE)

        model.load_state_dict(torch.load(CLF_PATH))

        return model