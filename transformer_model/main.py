import csv
from pycocoevalcap.bleu.bleu import Bleu
import sys, os
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from collections import Counter
import os
from PIL import Image
import string
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import pickle

import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

# ------------------------- TRAINING PARAMETERS -----------------------------
embed_size = 512
hidden_size = 512
num_layers = 6
english = 64
num_epochs = 1000
encoder_learning_rate = 1e-6
decoder_learning_rate = 1e-5
max_length = 30
freq_threshold = 5
patience = 6
best_bleu = -1
best_val_loss = float('inf')
no_improve = 0
encoder_linear_dropout_rate = 0.5
embed_dropout_rate = 0.5
decoder_dropout = 0.3
encoder_max_grad_clip_norm = 2.0
decoder_max_grad_clip_norm = 2.0

transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()

# ------------------------ EncoderCNN class ----------------------------
class EncoderCNN(nn.Module):
    def __init__(self, encoder_linear_dropout_rate, num_of_layers_to_unfreeze=-1, unfreeze_mode='top'):
        super().__init__()
        self.num_of_layers_to_unfreeze = num_of_layers_to_unfreeze
        self.unfreeze_mode = unfreeze_mode

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.dropout = nn.Dropout(p=encoder_linear_dropout_rate)

        # Identify parameterized layers (conv/bn layers)
        self.param_layers = [
            idx for idx, layer in enumerate(self.resnet)
            if len(list(layer.parameters())) > 0
        ]

        self.freeze_batchnorm()

        # Fine tune if num_of_layers_to_unfreeze is not -1
        if self.num_of_layers_to_unfreeze != -1:
            self.fine_tune()
        else:
            for p in self.resnet.parameters():
                p.requires_grad = True

    def forward(self, images):
        features = self.resnet(images)  # Shape: (batch_size, 2048, H, W)
        features = features.view(features.size(0), features.size(1), -1)  # Flatten spatial dimensions
        features = features.permute(0, 2, 1)  # Shape: (batch_size, H*W, 2048)
        features = self.dropout(features)
        return features

    def fine_tune(self):
        if self.num_of_layers_to_unfreeze <= 0 or self.num_of_layers_to_unfreeze > len(self.param_layers):
            raise ValueError(f"num_of_layers_to_unfreeze must be between 1 and {len(self.param_layers)}")

        # Freeze all first
        for p in self.resnet.parameters():
            p.requires_grad = False

        # Determine layers to unfreeze
        if self.num_of_layers_to_unfreeze > 0:
            if self.unfreeze_mode == 'top':
                layers = self.param_layers[-self.num_of_layers_to_unfreeze:]
            elif self.unfreeze_mode == 'bottom':
                layers = self.param_layers[:self.num_of_layers_to_unfreeze]
            else:
                raise ValueError("Use 'top' or 'bottom' mode")

            # Unfreeze selected layers
            for idx in layers:
                for param in self.resnet[idx].parameters():
                    param.requires_grad = True

    def freeze_batchnorm(self):
        for m in self.resnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

# ------------------------ TransformerDecoder class ----------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, vocab_size, num_layers=6, nhead=8, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, embed_size))

        # Linear layer to project features from 2048 to embed_size
        self.feature_projection = nn.Linear(2048, embed_size)

        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,  # Size of the embeddings
            nhead=nhead,         # Number of attention heads
            dim_feedforward=hidden_size,  # Hidden layer size in the feedforward network
            dropout=dropout      # Dropout rate
        )

        # Transformer Decoder
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final linear layer to predict the next word
        self.fc_out = nn.Linear(embed_size, vocab_size)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, features, captions):
        # features: (batch_size, num_pixels, 2048)
        # captions: (batch_size, caption_length)

        batch_size = features.size(0)
        caption_length = captions.size(1)

        # Project features from 2048 to embed_size
        features = self.feature_projection(features)  # (batch_size, num_pixels, embed_size)

        # Embed the captions
        embeddings = self.embed(captions)  # (batch_size, caption_length, embed_size)
        embeddings = embeddings + self.positional_encoding[:, :caption_length, :]

        # Transformer decoder expects (caption_length, batch_size, embed_size)
        embeddings = embeddings.permute(1, 0, 2)  # (caption_length, batch_size, embed_size)
        features = features.permute(1, 0, 2)  # (num_pixels, batch_size, embed_size)

        # Generate masks for the captions
        tgt_mask = self.generate_square_subsequent_mask(caption_length).to(features.device)
        tgt_padding_mask = (captions == self.vocab.stoi["<pad>"]).to(features.device)

        # Convert tgt_padding_mask to float with -inf and 0 values
        tgt_padding_mask = tgt_padding_mask.float().masked_fill(tgt_padding_mask == 1, float('-inf'))

        # Pass through the transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=embeddings,  # (caption_length, batch_size, embed_size)
            memory=features,  # (num_pixels, batch_size, embed_size)
            tgt_mask=tgt_mask,  # Mask to prevent attending to future tokens
            tgt_key_padding_mask=tgt_padding_mask  # Mask to ignore padding tokens
        )
        decoder_output = decoder_output.permute(1, 0, 2)  # (batch_size, caption_length, embed_size)

        # Predict the next word
        outputs = self.fc_out(decoder_output)  # (batch_size, caption_length, vocab_size)

        return outputs

    def generate(self, features, max_length=30):
        batch_size = features.size(0)
        captions = torch.full((batch_size, 1), self.vocab.stoi["<start>"], dtype=torch.long).to(features.device)

        # Project features from 2048 to embed_size
        features = self.feature_projection(features)  # (batch_size, num_pixels, embed_size)
        features = features.permute(1, 0, 2)  # (num_pixels, batch_size, embed_size)

        for _ in range(max_length):
            embeddings = self.embed(captions)  # (batch_size, current_length, embed_size)
            embeddings = embeddings + self.positional_encoding[:, :captions.size(1), :]
            embeddings = embeddings.permute(1, 0, 2)  # (current_length, batch_size, embed_size)

            tgt_mask = self.generate_square_subsequent_mask(captions.size(1)).to(features.device)
            tgt_padding_mask = (captions == self.vocab.stoi["<pad>"]).to(features.device)
            tgt_padding_mask = tgt_padding_mask.float().masked_fill(tgt_padding_mask == 1, float('-inf'))

            decoder_output = self.transformer_decoder(
                tgt=embeddings,  # (current_length, batch_size, embed_size)
                memory=features,  # (num_pixels, batch_size, embed_size)
                tgt_mask=tgt_mask,  # Mask to prevent attending to future tokens
                tgt_key_padding_mask=tgt_padding_mask  # Mask to ignore padding tokens
            )
            decoder_output = decoder_output.permute(1, 0, 2)  # (batch_size, current_length, embed_size)

            next_word_logits = self.fc_out(decoder_output[:, -1, :])  # (batch_size, vocab_size)
            next_word = next_word_logits.argmax(1).unsqueeze(1)

            captions = torch.cat([captions, next_word], dim=1)

            # Stop if all sequences have generated the <end> token
            if (next_word == self.vocab.stoi["<end>"]).all():
                break

        return captions

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# ------------------------- Vocabulary class --------------------------------
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        # Tokenize the text
        tokens = word_tokenize(text.lower())
        # Filter out punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        return tokens

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            tokens = self.tokenize(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokens = self.tokenize(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokens
        ]

# ---------------------- Dataset classes ----------------------------

# Training Dataset
class CocoDataset(Dataset):
    def __init__(self, root, annFile, vocab, transform=None, max_length=30, img_ids=None):
        self.root = root
        self.coco = COCO(annFile)
        self.img_ids = img_ids if img_ids else list(self.coco.imgs.keys())
        self.ann_ids = self.coco.getAnnIds(imgIds=self.img_ids)
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, index):
        ann_id = self.ann_ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])

        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        numerical_caption = [self.vocab.stoi["<start>"]]
        numerical_caption += self.vocab.numericalize(caption)
        numerical_caption.append(self.vocab.stoi["<end>"])

        if len(numerical_caption) > self.max_length:
            numerical_caption = numerical_caption[:self.max_length]
            numerical_caption[-1] = self.vocab.stoi["<end>"]
        else:
            numerical_caption += [self.vocab.stoi["<pad>"]] * (self.max_length - len(numerical_caption))

        return image, torch.tensor(numerical_caption)

# Validation Dataset (grouped by image_id)
class CocoValidationDataset(Dataset):
    def __init__(self, root, annFile, transform=None):
        self.root = root
        self.coco = COCO(annFile)
        self.img_ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns]

        return image, captions, img_id

# Modified Validation Dataset (returns individual caption-instance pairs)
class CocoValidationLossDataset(Dataset):
    def __init__(self, root, annFile, vocab, transform=None, max_length=30):
        self.root = root
        self.coco = COCO(annFile)
        self.ann_ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, index):
        ann_id = self.ann_ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root, img_info['file_name'])

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        numerical_caption = [self.vocab.stoi["<start>"]]
        numerical_caption += self.vocab.numericalize(caption)
        numerical_caption.append(self.vocab.stoi["<end>"])

        if len(numerical_caption) > self.max_length:
            numerical_caption = numerical_caption[:self.max_length]
            numerical_caption[-1] = self.vocab.stoi["<end>"]
        else:
            numerical_caption += [self.vocab.stoi["<pad>"]] * (self.max_length - len(numerical_caption))

        return image, torch.tensor(numerical_caption)

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    img_ids = [item[2] for item in batch]
    return images, captions, img_ids

# Test Dataset (20% split) - GROUPED BY IMAGE
class GroupedTestDataset(Dataset):
    def __init__(self, img_ids):
        self.coco = COCO('/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/annotations/captions_train2017.json')
        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        captions = [self.coco.anns[ann_id]['caption'] for ann_id in ann_ids]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join('/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/train2017/', img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        if transform:
            image = transform(image)
        return image, captions, img_id

# ------------------------------------ Main functions -----------------------------

def evaluate(encoder, decoder, loader, device):
    encoder.eval()
    decoder.eval()
    results = []
    references = {}  # Will store {img_id: [str, str...]}

    with torch.no_grad():
        for images, captions_list, img_ids in loader:
            images = images.to(device)
            features = encoder(images)
            generated = decoder.generate(features, max_length)

            # Process captions
            for i in range(generated.size(0)):
                img_id = img_ids[i]
                caption = generated[i].cpu().tolist()
                words = []
                for idx in caption:
                    word = decoder.vocab.itos.get(idx, "<unk>")
                    if word == "<end>": break
                    if word not in ["<start>", "<pad>"]:
                        words.append(word)
                results.append({"image_id": img_id, "caption": " ".join(words)})
                references[img_id] = captions_list[i]  # Store as list of strings

    # Convert to COCO format
    res = {item["image_id"]: [item["caption"]] for item in results}  # List of strings
    gts = references  # {img_id: [str, str...]}

    # Compute metrics
    bleu_scores = Bleu(4).compute_score(gts, res)

    return {
        'BLEU-4': bleu_scores[0][3],  # BLEU-4 score
    }

def train(encoder, decoder, train_loader, val_loss_loader, val_metrics_loader, device):
    global best_bleu
    global best_val_loss

    train_losses = []
    val_losses = []
    bleu_scores = []

    for epoch in range(num_epochs):
        # Training phase
        encoder.train()
        decoder.train()
        epoch_train_loss = 0

        print("\n Training started!\n")
        for idx, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            features = encoder(images)

            outputs = decoder(features, captions[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), encoder_max_grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), decoder_max_grad_clip_norm)
            optimizer.step()

            epoch_train_loss += loss.item() * images.size(0)

            if idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Calculate average training loss
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation phase
        encoder.eval()
        decoder.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            # Calculate validation loss
            print("\n Validation started!\n")
            for images, captions in val_loss_loader:
                images = images.to(device)
                captions = captions.to(device)

                features = encoder(images)
                outputs = decoder(features, captions[:, :-1])
                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
                epoch_val_loss += loss.item() * images.size(0)

        # Calculate average validation loss
        epoch_val_loss /= len(val_loss_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Calculate validation metrics
        print("\n Evaluating started!\n")
        val_metrics = evaluate(encoder, decoder, val_metrics_loader, device)
        bleu_scores.append(val_metrics['BLEU-4'])

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        print(f"BLEU-4: {val_metrics['BLEU-4']:.4f}\n")

        # Early stopping logic
        if val_metrics['BLEU-4'] > best_bleu or epoch_val_loss < best_val_loss:
            if val_metrics['BLEU-4'] > best_bleu:
                best_bleu = val_metrics['BLEU-4']

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss

            no_improve = 0
            torch.save(encoder.state_dict(), 'best_encoder.pth')
            torch.save(decoder.state_dict(), 'best_decoder.pth')

            # torch.save(encoder_scheduler.state_dict(), "encoder_scheduler.pth")
            # torch.save(decoder_scheduler.state_dict(), "decoder_scheduler.pth")

            torch.save(optimizer.state_dict(), "optimizer.pth")
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered!")
            break

        # # Scheduler
        # encoder_scheduler.step(epoch_val_loss)
        # decoder_scheduler.step(epoch_val_loss)

    return train_losses, val_losses, bleu_scores

def save_training_metrics(train_losses, val_losses, bleu_scores):
    # Combine data into rows
    epochs = list(range(1, len(train_losses) + 1))
    rows = zip(epochs, train_losses, val_losses, bleu_scores)

    # Save to CSV file
    with open("training_results.csv", "w", newline="") as f:
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
    plt.savefig('training_metrics.png')
    plt.close()

def test(encoder, decoder, test_loader, device):
    print('\n Testing started!')
    # Load best model checkpoint if using early stopping
    encoder.load_state_dict(torch.load('best_encoder.pth'))
    decoder.load_state_dict(torch.load('best_decoder.pth'))

    # Final evaluation
    final_metrics = evaluate(encoder, decoder, test_loader, device)

    with open('test.log', 'w') as f:
        f.write(str(final_metrics['BLEU-4']))
        f.write('\n')

    print("\nFinal Test Metrics:")
    print(f"BLEU-4: {final_metrics['BLEU-4']:.4f}")

if __name__ == "__main__":

    # -------------------------- DEVICE -------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------- DATASETS / DATALOADERS ------------------------

    # Prepare
    # Load full dataset
    full_coco = COCO('/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/annotations/captions_train2017.json')
    all_img_ids = np.array(full_coco.getImgIds())

    # Split 80-20 (vc there is no testing data so taking it from train)
    np.random.seed(42)
    np.random.shuffle(all_img_ids)
    split_idx = int(0.8 * len(all_img_ids))
    train_img_ids = all_img_ids[:split_idx]
    test_img_ids = all_img_ids[split_idx:]

    # Build vocabulary using ONLY training captions
    train_ann_ids = full_coco.getAnnIds(imgIds=train_img_ids)
    train_captions = [full_coco.anns[ann_id]['caption'] for ann_id in train_ann_ids]

    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(train_captions)
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print(f'Vocab length: {len(vocab)}')

    # Transform
    transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()

    # Train
    # Training Dataset (80%)
    train_dataset = CocoDataset(
        root='/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/train2017/',
        annFile='/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/annotations/captions_train2017.json',
        vocab=vocab,
        transform=transform,
        max_length=max_length,
        img_ids=train_img_ids.tolist()
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Validation
    # For validation LOSS (individual caption-instance pairs)
    val_loss_loader = DataLoader(
        CocoValidationLossDataset(
            root='/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/val2017/',
            annFile='/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/annotations/captions_val2017.json',
            vocab=vocab,
            transform=transform,
            max_length=max_length
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # For validation METRICS (grouped by image)
    val_metrics_loader = DataLoader(
        CocoValidationDataset(  # Your existing grouped dataset class
            root='/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/val2017/',
            annFile='/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/annotations/captions_val2017.json',
            transform=transform
        ),
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    # Test
    test_dataset = GroupedTestDataset(test_img_ids.tolist())
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=lambda x: (torch.stack([xi[0] for xi in x]), [xi[1] for xi in x], [xi[2] for xi in x])
    )

    # Saving for later use
    torch.save(test_loader.dataset, "test_dataset.pth")

    # ------------------------ MODEL DEFINITION ------------------------------

    # Encoder
    encoder = EncoderCNN(encoder_linear_dropout_rate, num_of_layers_to_unfreeze=6).to(device)

    # Decoder
    decoder = TransformerDecoder(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab=vocab,
        vocab_size=len(vocab),
        num_layers=6,
        nhead=8,
        dropout=decoder_dropout
    ).to(device)

    # Criterion
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': encoder_learning_rate},
        {'params': decoder.parameters(), 'lr': decoder_learning_rate}
    ], weight_decay=1e-4)

    # ------------------------ TRAIN ------------------------------
    sys.path.append('/home/obojana/bojana/pycocoevalcap')
    os.environ["METEOR_JAR"] = "/home/obojana/bojana/pycocoevalcap/meteor/meteor-1.5.jar"

    train_losses, val_losses, bleu_scores = train(encoder, decoder, train_loader, val_loss_loader, val_metrics_loader, device)

    # ------------------------ SAVE and PLOT ------------------------------
    save_training_metrics(train_losses, val_losses, bleu_scores)
    plot_training_metrics(train_losses, val_losses, bleu_scores)

    # ------------------------ TEST ------------------------------
    test(encoder, decoder, test_loader, device)
