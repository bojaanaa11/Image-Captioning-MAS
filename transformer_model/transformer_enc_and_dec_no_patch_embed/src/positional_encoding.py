import torch
import torch.nn as nn
import src.parameters as params

class PositionEncoding(nn.Module):
    def __init__(self, d_model=params.embed_size, max_len=params.max_length * 2):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        embedding_index = torch.arange(0, d_model, 2, dtype=torch.float)

        div_term = torch.exp(embedding_index * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(1), :].unsqueeze(0)