import sys, os

import torch
import torch.nn as nn

from src.cnn_encoder import EncoderCNN
from src.transformer_decoder import TransformerDecoder

from src.train import train

from src.prepare import prepare
import src.parameters as params

if __name__ == "__main__":
    train_loader, val_loss_loader, val_metrics_loader, test_loader, vocab = prepare()

    # Encoder
    encoder = EncoderCNN(
        params.cnn_encoder_linear_dropout_rate,
        num_of_layers_to_unfreeze=params.cnn_encoder_num_layers
    ).to(params.device)

    # Decoder
    decoder = TransformerDecoder(
        embed_size=params.embed_size,
        hidden_size=params.hidden_size,
        vocab=vocab,
        vocab_size=len(vocab),
        num_layers=params.transformer_decoder_num_layers,
        nhead=params.transfomer_decoder_nhead,
        dropout=params.transformer_decoder_dropout
    ).to(params.device)

    # Criterion
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])

    # Optimizer
    # Get the parameters for each part of the model
    encoder_params = list(encoder.parameters())
    transformer_encoder_params = list(encoder.transformer_encoder.parameters())

    # Remove transformer_encoder parameters from encoder_params to avoid duplication
    encoder_params = [param for param in encoder_params if id(param) not in {id(p) for p in transformer_encoder_params}]

    # Define the optimizer with different learning rates for each part
    optimizer = torch.optim.Adam([
        {'params': encoder_params, 'lr': params.cnn_encoder_learning_rate},  # For the main encoder
        {'params': decoder.parameters(), 'lr': params.transformer_decoder_learning_rate},  # For the decoder
        {'params': transformer_encoder_params, 'lr': params.transformer_encoder_learning_rate}  # For the transformer encoder part
    ], weight_decay=params.weight_decay)

    # ------------------------ TRAIN ------------------------------
    sys.path.append('/home/obojana/bojana/pycocoevalcap')
    os.environ["METEOR_JAR"] = "/home/obojana/bojana/pycocoevalcap/meteor/meteor-1.5.jar"

    train(
        encoder,
        decoder,
        criterion,
        optimizer,
        train_loader,
        val_loss_loader,
        val_metrics_loader,
        params.device,
        params.num_epochs,
        params.cnn_encoder_max_grad_clip_norm,
        params.transformer_decoder_max_grad_clip_norm,
        params.patience
    )