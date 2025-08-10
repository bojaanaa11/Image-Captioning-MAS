import csv
import matplotlib.pyplot as plt

def save_training_metrics(train_losses, val_losses, bleu_scores):
    # Combine data into rows
    epochs = list(range(1, len(train_losses) + 1))
    rows = zip(epochs, train_losses, val_losses, bleu_scores)

    # Save to CSV file
    with open("/home/obojana/bojana/src_saved/training_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "BLEU-4"])  # Header
        writer.writerows(rows)

    plot_training_metrics(train_losses, val_losses, bleu_scores)

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