import csv
import json
import sys, os
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import nltk
from nltk.tokenize import word_tokenize
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
import datetime
import os
import subprocess
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# ------------------------- TRAINING PARAMETERS -----------------------------

if len(sys.argv) < 2:
    exit('Please provide the starting epoch number as a command line argument!')
start_epoch = int(sys.argv[1])
num_epochs = 1000

num_workers = 0

batch_size = 32

cnn_encoder_learning_rate = 1e-6
transformer_decoder_learning_rate = 1e-6

cnn_num_of_layers_to_unfreeze = 6

beam_size = 12

transformer_encoder_embed_size = 512
transformer_encoder_num_layers = 6
transformer_encoder_nhead = 8

transformer_decoder_embed_size = 512
transformer_decoder_hidden_size = 512
transformer_decoder_num_layers = 6
transformer_decoder_nhead = 8
transformer_decoder_dropout = 0.1

weight_decay = 1e-4

freq_threshold = 5

no_improve = 0
patience = 6

best_bleu = -1
best_val_loss = float('inf')

max_length = 20

cnn_encoder_max_grad_clip_norm = 2.0
transformer_decoder_max_grad_clip_norm = 2.0

transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()

# ================================================================================================================================================================================================ HELPER FUNCTIONS

def download_coco_dataset():
    global dataset_dir

    dataset_dir = os.path.expanduser(
        "/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/"
    )
    annotations_path = os.path.join(dataset_dir, "annotations/captions_train2017.json")

    if os.path.exists(annotations_path):
        return

    print("COCO dataset not found. Downloading...")

    try:
        subprocess.run(["pip", "install", "kaggle"], check=True)

        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "awsaf49/coco-2017-dataset"],
            check=True
        )

        os.makedirs(dataset_dir, exist_ok=True)
        subprocess.run(
            ["unzip", "coco-2017-dataset.zip", "-d", dataset_dir],
            check=True
        )
        print("Dataset downloaded successfully!")

    except Exception as e:
        print(f"Download failed: {e}")
        print("Manual steps:")
        print("1. Install Kaggle CLI: pip install kaggle")
        print("2. Configure API key: https://github.com/Kaggle/kaggle-api#api-credentials")
        print("3. Download dataset: kaggle datasets download -d awsaf49/coco-2017-dataset")
        print(f"4. Unzip to: {dataset_dir}")
        exit()

# ================================================================================================================================================================================================ MODEL CLASSES

# -------------------------------------------------------- EncoderCNN class --------------------------------------------------------------
class EncoderCNN(nn.Module):
    def __init__(self, num_of_layers_to_unfreeze=-1, unfreeze_mode='top'):
        super().__init__()
        self.num_of_layers_to_unfreeze = num_of_layers_to_unfreeze
        self.unfreeze_mode = unfreeze_mode

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

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
                p.requires_grad = False

    def forward(self, images):
        features = self.resnet(images)  # (batch_size, 2048, 7, 7)
        features = features.permute(0, 2, 3, 1)  # (batch_size, 7, 7, 2048)
        return features  # Direct spatial features without flattening

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

# -------------------------------------------------------- TransformerImageEncoder class --------------------------------------------------------------

class TransformerImageEncoder(nn.Module):
    def __init__(self, embed_size, num_layers=6, nhead=8):
        super().__init__()
        self.encoder_layer = TransformerEncoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=embed_size * 4,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, x):
        return self.transformer_encoder(x)

# -------------------------------------------------------- TransformerCaptionDecoder class --------------------------------------------------------------

class TransformerCaptionDecoder(nn.Module):
    def __init__(self, transformer_encoder, embed_size, hidden_size, vocab, vocab_size, max_length, num_layers=6, nhead=8, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout

        self.image_encoder = transformer_encoder

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
            dropout=dropout,      # Dropout rate
            batch_first=False
        )

        # Transformer Decoder
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Final linear layer to predict the next word
        self.fc_out = nn.Linear(embed_size, vocab_size)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, features, captions):
        # Project features through patch embedding and transformer encoder
        features = self.feature_projection(features)  # (batch_size, 7, 7, 512)
        features = features.flatten(1, 2)              # (batch_size, 49, 512)
        features = self.image_encoder(features)         # (batch_size, 49, 512)
        features = features.permute(1, 0, 2)            # (49, batch_size, 512) <-- ADD THIS

        caption_length = captions.size(1)

        # Embed the captions
        embeddings = self.embed(captions)  # (batch_size, caption_length, embed_size)
        embeddings = embeddings + self.positional_encoding[:, :caption_length, :]

        # Transformer decoder expects (caption_length, batch_size, embed_size)
        embeddings = embeddings.permute(1, 0, 2)  # (caption_length, batch_size, embed_size)

        # Generate masks for the captions
        tgt_mask = self.generate_square_subsequent_mask(caption_length).to(features.device)
        tgt_padding_mask = (captions == self.vocab.stoi["<pad>"]).to(features.device)

        # Convert tgt_padding_mask to float with -inf and 0 values
        # tgt_padding_mask = tgt_padding_mask.float().masked_fill(tgt_padding_mask == 1, float('-inf'))

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

    def generate(self, features):
        features = self.feature_projection(features)  # (batch_size, 7, 7, 512)
        features = features.flatten(1, 2)              # (batch_size, 49, 512)
        features = self.image_encoder(features)         # (batch_size, 49, 512)
        features = features.permute(1, 0, 2)            # (49, batch_size, 512) <-- ADD THIS

        batch_size = features.size(1)

        captions = torch.full((batch_size, 1), self.vocab.stoi["<start>"], dtype=torch.long).to(features.device)

        for _ in range(self.max_length):
            embeddings = self.embed(captions)  # (batch_size, current_length, embed_size)
            embeddings = embeddings + self.positional_encoding[:, :captions.size(1), :]
            embeddings = embeddings.permute(1, 0, 2)  # (current_length, batch_size, embed_size)

            tgt_mask = self.generate_square_subsequent_mask(captions.size(1)).to(features.device)
            tgt_padding_mask = (captions == self.vocab.stoi["<pad>"]).to(features.device)
            # tgt_padding_mask = tgt_padding_mask.float().masked_fill(tgt_padding_mask == 1, float('-inf'))

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

            # Stop if any sequences have generated the <end> token
            if (next_word == self.vocab.stoi["<end>"]).any():
                break

        return captions

    def generate_beam(self, features, beam_size):
        # Process image features (same as before)
        features = self.feature_projection(features)  # (batch_size, 7, 7, 512)
        features = features.flatten(1, 2)             # (batch_size, 49, 512)
        features = self.image_encoder(features)       # (batch_size, 49, 512)
        features = features.permute(1, 0, 2)          # (49, batch_size, 512)
        batch_size = features.size(1)

        # Container for final sequences
        all_seqs = []

        # Process each image in the batch separately
        for i in range(batch_size):
            # Extract features for the i-th image
            img_feature = features[:, i:i+1]  # (49, 1, 512)

            # Initialize beam: (sequence, score)
            beams = [([self.vocab.stoi["<start>"]], 0.0)]
            finished = []

            # Beam search for max_length steps
            for step in range(self.max_length):
                new_beams = []

                # Expand each active beam
                for seq, score in beams:
                    # Skip finished sequences
                    if seq[-1] == self.vocab.stoi["<end>"]:
                        finished.append((seq, score))
                        continue

                    # Prepare input tensor for the beam
                    seq_tensor = torch.tensor(seq, device=features.device).unsqueeze(0)  # (1, seq_len)

                    # Embed sequence and add positional encoding
                    embeddings = self.embed(seq_tensor)  # (1, seq_len, embed_size)
                    embeddings = embeddings + self.positional_encoding[:, :seq_tensor.size(1), :]
                    embeddings = embeddings.permute(1, 0, 2)  # (seq_len, 1, embed_size)

                    # Generate masks
                    tgt_mask = self.generate_square_subsequent_mask(seq_tensor.size(1)).to(features.device)
                    padding_mask = torch.zeros_like(seq_tensor).bool().to(features.device)

                    # Decode
                    decoder_out = self.transformer_decoder(
                        tgt=embeddings,
                        memory=img_feature,
                        tgt_mask=tgt_mask,
                        tgt_key_padding_mask=padding_mask
                    )  # (seq_len, 1, embed_size)

                    # Predict next token (last token only)
                    last_output = decoder_out[-1, :, :]  # (1, embed_size)
                    logits = self.fc_out(last_output)    # (1, vocab_size)
                    log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (vocab_size)

                    # Get top candidates for this beam
                    topk_log_probs, topk_ids = log_probs.topk(beam_size)

                    # Create new beams
                    for j in range(beam_size):
                        token_id = topk_ids[j].item()
                        new_score = score + topk_log_probs[j].item()
                        new_seq = seq + [token_id]
                        new_beams.append((new_seq, new_score))

                # Select top beams overall
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

                # Early stop if all beams are finished
                if all(seq[-1] == self.vocab.stoi["<end>"] for seq, _ in beams):
                    finished += beams
                    beams = []
                    break

            # Collect finished and active beams
            candidates = finished + beams
            if not candidates:
                # Fallback: use start token only
                best_seq = [self.vocab.stoi["<start>"], self.vocab.stoi["<end>"]]
            else:
                # Select best sequence by score
                best_seq = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
            all_seqs.append(best_seq)

        # Pad sequences to max length in the batch
        max_len = max(len(seq) for seq in all_seqs)
        padded_seqs = torch.full((batch_size, max_len), self.vocab.stoi["<pad>"], device=features.device)
        for i, seq in enumerate(all_seqs):
            padded_seqs[i, :len(seq)] = torch.tensor(seq, device=features.device)

        return padded_seqs

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# ================================================================================================================================================================================================ DATASET AND VOCABULARY CLASSES

# -------------------------------------------------------- Vocabulary class --------------------------------------------------------------

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
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        nltk.data.find('tokenizers/punkt')

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
        if not text.strip():
            return [self.stoi["<unk>"]]
        tokens = self.tokenize(text)
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokens
        ]

# -------------------------------------------------------- CocoDataset class - Training --------------------------------------------------------------

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

        try:
            img_path = os.path.join(self.root, img_info['file_name'])
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Corrupted image skipped: {img_path}")
            return torch.zeros(3, 224, 224), torch.full((self.max_length,), self.vocab.stoi["<pad>"])

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

# -------------------------------------------------------- CocoValidationDataset class - Validation Grouped By image_id --------------------------------------------------------------

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

# -------------------------------------------------------- CocoValidationDataset class - Validation --------------------------------------------------------------

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

# Used for grouped by image_id validation dataset
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    img_ids = [item[2] for item in batch]
    return images, captions, img_ids

# -------------------------------------------------------- GroupedTestDataset class - Test (10% Training) --------------------------------------------------------------

class GroupedTestDataset(Dataset):
    def __init__(self, img_ids):
        self.coco = COCO(os.path.join(dataset_dir, 'annotations/captions_train2017.json'))
        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        captions = [self.coco.anns[ann_id]['caption'] for ann_id in ann_ids]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(os.path.join(dataset_dir, 'train2017/'), img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        if transform:
            image = transform(image)
        return image, captions, img_id

# ================================================================================================================================================================================================ MAIN FLOW

def evaluate(loader):
    cnn_encoder.eval()
    transformer_decoder.eval()
    results = []
    references = {}

    # Prepare tokenizer once
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    with torch.no_grad():
        for images, captions_list, img_ids in loader:
            images = images.to(device)
            features = cnn_encoder(images)
            generated = transformer_decoder.generate_beam(features, beam_size)

            for i in range(generated.size(0)):
                img_id = img_ids[i]
                caption = generated[i].cpu().tolist()
                words = []
                for idx in caption:
                    word = transformer_decoder.vocab.itos.get(idx, "<unk>")
                    if word == "<end>":
                        break
                    if word not in ["<start>", "<pad>"]:
                        words.append(word)
                generated_caption = " ".join(words)
                results.append({"image_id": img_id, "caption": generated_caption})
                references[img_id] = captions_list[i]

    # Prepare data for BLEU calculation
    hypotheses = []
    ref_list = []

    for res in results:
        img_id = res["image_id"]
        # Tokenize generated caption
        hyp_tokens = word_tokenize(res["caption"].lower())
        hypotheses.append(hyp_tokens)

        # Tokenize all reference captions
        refs_for_image = []
        for ref in references[img_id]:
            ref_tokens = word_tokenize(ref.lower())
            refs_for_image.append(ref_tokens)
        ref_list.append(refs_for_image)

    # Calculate BLEU-4 with smoothing
    smoothie = SmoothingFunction().method4
    bleu4 = corpus_bleu(
        ref_list,
        hypotheses,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothie
    )

    return {'BLEU-4': bleu4}

def train():
    global best_bleu
    global best_val_loss
    global no_improve
    global checkpoints_dir

    if start_epoch > 0:
        checkpoint = torch.load(os.path.join(checkpoints_dir, f'epoch{start_epoch-1}_FULL.pth'))
        cnn_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        transformer_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_bleu = checkpoint['bleu']
        best_val_loss = checkpoint['val_loss']

    checkpoint_freq = len(train_loader) // 3

    f = open(training_results_csv_path, 'a+')
    writer = csv.writer(f)
    if start_epoch == 0:
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "BLEU-4"])

    for epoch in range(start_epoch, num_epochs):
        # Training phase
        cnn_encoder.train()
        transformer_decoder.train()

        epoch_train_loss = 0.

        print(">>>> Training started! <<<<\n")
        for idx, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            features = cnn_encoder(images)

            outputs = transformer_decoder(features, captions[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnn_encoder.parameters(), cnn_encoder_max_grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(transformer_decoder.parameters(), transformer_decoder_max_grad_clip_norm)
            optimizer.step()

            epoch_train_loss += loss.item() * images.size(0)

            if idx % 100 == 0:
                torch.cuda.empty_cache()
                print(f'Epoch [{epoch}/{num_epochs}], Step [{idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

            if idx % checkpoint_freq == 0 and idx > 0:
                # torch.save({
                #     'epoch': epoch,
                #     'batch_idx': idx,
                #     'encoder_state_dict': cnn_encoder.state_dict(),
                #     'decoder_state_dict': transformer_decoder.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'loss': loss.item(),
                # }, os.path.join(checkpoints_dir, f'checkpoint_epoch{epoch}_batch{idx}.pth'))

                torch.cuda.empty_cache()

        # Calculate average training loss
        epoch_train_loss /= len(train_loader.dataset)

        # Validation phase
        cnn_encoder.eval()
        transformer_decoder.eval()

        epoch_val_loss = 0.
        with torch.no_grad():
            # Calculate validation loss
            print(">>> Validation started! <<< \n")
            for images, captions in val_loss_loader:
                images = images.to(device)
                captions = captions.to(device)

                features = cnn_encoder(images)
                outputs = transformer_decoder(features, captions[:, :-1])
                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
                epoch_val_loss += loss.item() * images.size(0)

        # Calculate average validation loss
        epoch_val_loss /= len(val_loss_loader.dataset)

        # Calculate validation metrics
        print(">>>> Evaluating started! <<<<\n")
        val_metrics = evaluate(val_metrics_loader)
        bleu4 = val_metrics['BLEU-4']

        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} \n")
        print(f"BLEU-4: {val_metrics['BLEU-4']:.4f} \n")

        writer.writerow([epoch, epoch_train_loss, epoch_val_loss, bleu4])
        f.flush()

        # Early stopping logic
        if val_metrics['BLEU-4'] > best_bleu or epoch_val_loss < best_val_loss:
            if val_metrics['BLEU-4'] > best_bleu:
                best_bleu = val_metrics['BLEU-4']

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss

            no_improve = 0
            torch.save({
                'epoch': epoch,
                'batch_idx': idx,
                'encoder_state_dict': cnn_encoder.state_dict(),
                'decoder_state_dict': transformer_decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': {'bleu': val_metrics['BLEU-4'], 'val_loss': epoch_val_loss}
            }, os.path.join(base_dir, 'best_model.pth'))
        else:
            no_improve += 1
            print(f'No improve: {no_improve}')

        if no_improve >= patience:
            print("Early stopping triggered!")
            break

        torch.save({
            'epoch': epoch,
            'batch_idx': len(train_loader)-1,  # Mark as end of epoch
            'encoder_state_dict': cnn_encoder.state_dict(),
            'decoder_state_dict': transformer_decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'bleu': val_metrics['BLEU-4'],
        }, os.path.join(checkpoints_dir, f'epoch{epoch}_FULL.pth'))

    f.close()

def test():
    print('>>>> Testing started! <<<<\n')
    # Load best model checkpoint if using early stopping
    checkpoint = torch.load(os.path.join(base_dir, 'best_model.pth'))
    cnn_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    transformer_decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # Final evaluation
    final_metrics = evaluate(test_loader)

    with open(test_results_log_path, 'w') as f:
        f.write(str(final_metrics['BLEU-4']))
        f.write('\n')

    print(">>>> Final Test Metrics: <<<<\n")
    print(f"BLEU-4: {final_metrics['BLEU-4']:.4f}")

# ================================================================================================================================================================================================  SAVING AND PLOTING

def plot_training_metrics():
    with open(training_results_csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        train_losses = []
        val_losses = []
        bleu_scores = []
        for row in reader:
            train_losses.append(float(row[1]))
            val_losses.append(float(row[2]))
            bleu_scores.append(float(row[3]))
    if not train_losses or not val_losses or not bleu_scores:
        print("No training metrics found. Skipping plotting.")
        return
    print("Plotting training metrics...")

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
    plt.savefig(training_metrics_plot_path)
    plt.close()

if __name__ == "__main__":

    # --------------------------------------------------------------- DOWNLOAD DATASET -------------------------------------------------------------

    dataset_dir = ''
    download_coco_dataset()

    # ------------------------------------------------------------ DIRECTORIES AND FILES ---------------------------------------------------------------

    if start_epoch == 0:
        base_dir = '/home/obojana/bojana/transformer_training_stats_' + '_'.join(str(datetime.datetime.now()).split())
    else:
        if len(sys.argv) < 3:
            exit('Please provide the path of base directory!')

        base_dir = sys.argv[2]

    print(f'Base directory for training stats: {base_dir}')

    checkpoints_dir = os.path.join(base_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f'Directory for saving checkpoints: {checkpoints_dir}')

    train_split_path = os.path.join(base_dir, 'train_split_img_ids.json')
    test_split_path = os.path.join(base_dir, 'test_split_img_ids.json')
    print(f'Train split path: {train_split_path}')
    print(f'Test split path: {test_split_path}')

    vocab_path = os.path.join(base_dir, 'vocab.pkl')
    print(f'Vocabulary path: {vocab_path}')

    sys.path.append('/home/obojana/bojana/pycocoevalcap')
    os.environ["METEOR_JAR"] = "/home/obojana/bojana/pycocoevalcap/meteor/meteor-1.5.jar"

    training_results_csv_path = os.path.join(base_dir, 'training_results.csv')

    test_results_log_path = os.path.join(base_dir, 'test_results.log')

    training_metrics_plot_path = os.path.join(base_dir, 'training_metrics.png')

    # --------------------------------------------------------------------- DEVICE ---------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------------------------------------------------- VOCABULARY & DATASETS / DATALOADERS ---------------------------------------------------

    # Prepare
    # Load full dataset
    try:
        full_coco = COCO(os.path.join(dataset_dir, 'annotations/captions_train2017.json'))
        print("Annotations loaded successfully!")
    except Exception as e:
        print(f"Error loading annotations: {e}")
        print("Please ensure the COCO dataset is downloaded and the path is correct.")
        exit()

    if start_epoch == 0:
        # Get all image IDs
        all_img_ids = np.array(full_coco.getImgIds())

        # Split 80-20 (vc there is no testing data so taking it from train)
        np.random.seed(42)
        np.random.shuffle(all_img_ids)
        split_idx = int(0.8 * len(all_img_ids))
        train_img_ids = all_img_ids[:split_idx]
        test_img_ids = all_img_ids[split_idx:]

        # Saving for later use
        with open(train_split_path, "w") as f:
            json.dump(train_img_ids.tolist(), f)

        with open(test_split_path, "w") as f:
            json.dump(test_img_ids.tolist(), f)
    else:

        with open(train_split_path) as f:
            train_img_ids = np.array(json.load(f))

        with open(test_split_path) as f:
            test_img_ids = np.array(json.load(f))

    # ---------------------------- VOCABULARY -------------------------------

    if start_epoch == 0:
        print("Building vocabulary...")

        train_ann_ids = full_coco.getAnnIds(imgIds=train_img_ids)
        train_captions = [full_coco.anns[ann_id]['caption'] for ann_id in train_ann_ids]

        vocab = Vocabulary(freq_threshold)
        vocab.build_vocabulary(train_captions)
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
    else:
        print("Loading existing vocabulary...")
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)
    print(f'Vocab length: {len(vocab)}')

    # ----------------------------------- DATASETS & DATALOADERS ------------------------------

    # Train
    # Training Dataset (90%)
    train_dataset = CocoDataset(
        root=os.path.join(dataset_dir, 'train2017/'),
        annFile=os.path.join(dataset_dir, 'annotations/captions_train2017.json'),
        vocab=vocab,
        transform=transform,
        max_length=max_length,
        img_ids=train_img_ids.tolist()
    )

    # Test
    test_dataset = GroupedTestDataset(test_img_ids.tolist())

    # Train Loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Test Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=lambda x: (torch.stack([xi[0] for xi in x]), [xi[1] for xi in x], [xi[2] for xi in x]),
        num_workers=num_workers
    )

    # Validation
    # For validation LOSS (individual caption-instance pairs)
    val_loss_loader = DataLoader(
        CocoValidationLossDataset(
            root=os.path.join(dataset_dir, 'val2017/'),
            annFile=os.path.join(dataset_dir, 'annotations/captions_val2017.json'),
            vocab=vocab,
            transform=transform,
            max_length=max_length
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # For validation METRICS (grouped by image)
    val_metrics_loader = DataLoader(
        CocoValidationDataset(  # Your existing grouped dataset class
            root=os.path.join(dataset_dir, 'val2017/'),
            annFile=os.path.join(dataset_dir, 'annotations/captions_val2017.json'),
            transform=transform
        ),
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers
    )

    # ------------------------------------------------------------------ MODEL DEFINITION ---------------------------------------------------------

    # CNN Encoder
    cnn_encoder = EncoderCNN(
        num_of_layers_to_unfreeze=cnn_num_of_layers_to_unfreeze
    )
    cnn_encoder.to(device)

    # Transformer Encoder
    transformer_encoder = TransformerImageEncoder(
        embed_size=transformer_encoder_embed_size,
        num_layers=transformer_encoder_num_layers,
        nhead=transformer_encoder_nhead
    )

    # Transformer Decoder
    transformer_decoder = TransformerCaptionDecoder(
        transformer_encoder=transformer_encoder,
        embed_size=transformer_decoder_embed_size,
        hidden_size=transformer_decoder_hidden_size,
        vocab=vocab,
        vocab_size=len(vocab),
        max_length=max_length,
        num_layers=transformer_decoder_num_layers,
        nhead=transformer_decoder_nhead,
        dropout=transformer_decoder_dropout
    )
    transformer_decoder.to(device)

    n_cnn_parameters = sum(p.numel() for p in cnn_encoder.parameters() if p.requires_grad)
    n_tr_parameters = sum(p.numel() for p in transformer_decoder.parameters() if p.requires_grad)
    print(f"cnn Number of params: {n_cnn_parameters}")
    print(f"tr Number of params: {n_tr_parameters}")

    # Criterion
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': cnn_encoder.parameters(), 'lr': cnn_encoder_learning_rate},
        {'params': transformer_decoder.parameters(), 'lr': transformer_decoder_learning_rate}
    ], weight_decay=weight_decay)

    # ------------------------------------------------------- TRAIN ------------------------------------------------------------

    train()

    # -------------------------------------------------------  PLOT TRAINING METRICS -----------------------------------------------------------
    plot_training_metrics()

    # ------------------------------------------------------------ TEST ---------------------------------------------------------------
    test()
