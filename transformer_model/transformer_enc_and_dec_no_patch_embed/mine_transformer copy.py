import csv
import sys, os

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from src.cnn_encoder import EncoderCNN
from src.transformer_decoder import TransformerDecoder
from src.eval import evaluate
from src.train import train

from src.prepare import prepare
import src.parameters as params


def save_training_metrics(train_losses, val_losses, bleu_scores):
    # Combine data into rows
    epochs = list(range(1, len(train_losses) + 1))
    rows = zip(epochs, train_losses, val_losses, bleu_scores)

    # Save to CSV file
    with open("/home/obojana/bojana/src_saved/training_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "BLEU-4"])  # Header
        writer.writerows(rows)

def plot_training_metrics(train_losses, val_losses, bleu_scores):
    plt.figure(figsize=(12, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker = 's')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.xticks(range(1, len(epochs) + 1))
    plt.ylabel('Loss')
    plt.legend()

    # Metrics plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, bleu_scores, label='BLEU-4', marker='^')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.xticks(range(1, len(epochs) + 1))
    plt.legend()

    plt.tight_layout()
    plt.savefig('/home/obojana/bojana/src_saved/training_metrics.png')
    plt.close()

def test(encoder, decoder, test_loader, device):
    print('\n Testing started!')
    # Load best model checkpoint if using early stopping
    encoder.load_state_dict(torch.load('/home/obojana/bojana/src_saved/best_encoder.pth'))
    decoder.load_state_dict(torch.load('/home/obojana/bojana/src_saved/best_decoder.pth'))

    # Final evaluation
    final_metrics = evaluate(encoder, decoder, test_loader, device)

    with open('test.log', 'w') as f:
        f.write(str(final_metrics['BLEU-4']))
        f.write('\n')

    print("\nFinal Test Metrics:")
    print(f"BLEU-4: {final_metrics['BLEU-4']:.4f}")

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

    train_losses, val_losses, bleu_scores = train(
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

    # ------------------------ SAVE and PLOT ------------------------------
    save_training_metrics(train_losses, val_losses, bleu_scores)
    plot_training_metrics(train_losses, val_losses, bleu_scores)

    # ------------------------ TEST ------------------------------
    test(encoder, decoder, test_loader, params.device)