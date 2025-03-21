import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, hidden_size, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,  # Size of the embeddings
            nhead=num_heads,     # Number of attention heads
            dim_feedforward=hidden_size,  # Hidden layer size in the feedforward network
            dropout=dropout      # Dropout rate
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        # src: (sequence_length, batch_size, embed_size)
        return self.transformer_encoder(src)