import torch
from mine_att import EncoderCNN, DecoderRNN, Vocabulary, GroupedTestDataset, freq_threshold, encoder_linear_dropout_rate, embed_dropout_rate, lstm_dropout_rate, num_layers, embed_size, hidden_size
import numpy as np
from pycocotools.coco import COCO
from torchvision import transforms, models
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset and vocabulary
full_coco = COCO('/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/annotations/captions_train2017.json')
all_img_ids = np.array(full_coco.getImgIds())

np.random.seed(42)
np.random.shuffle(all_img_ids)
split_idx = int(0.8 * len(all_img_ids))
train_img_ids = all_img_ids[:split_idx]
test_img_ids = all_img_ids[split_idx:]

train_ann_ids = full_coco.getAnnIds(imgIds=train_img_ids)
train_captions = [full_coco.anns[ann_id]['caption'] for ann_id in train_ann_ids]

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()

# Load test dataset
test_dataset = torch.load("test_dataset.pth")
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    collate_fn=lambda x: (torch.stack([xi[0] for xi in x]), [xi[1] for xi in x], [xi[2] for xi in x])
)

# Load trained models
encoder = EncoderCNN(encoder_linear_dropout_rate, num_of_layers_to_unfreeze=6).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab, len(vocab), embed_dropout_rate, lstm_dropout_rate, num_layers).to(device)

encoder.load_state_dict(torch.load('best_encoder.pth'))
decoder.load_state_dict(torch.load('best_decoder.pth'))

encoder.eval()
decoder.eval()

# Select a random instance from the test set
random_index = random.randint(0, len(test_loader.dataset) - 1)
for i, (image, true_captions, img_id) in enumerate(test_loader):
    if i == random_index:
        break

# Generate caption and attention weights
with torch.no_grad():
    image = image.to(device)
    features = encoder(image)
    captions, attention_weights = decoder.generate_beam_with_attention(features, 10, max_length=30)

# Convert caption indices to words
caption_words = []
for idx in captions[0].cpu().numpy():
    word = vocab.itos.get(idx, "<unk>")
    if word == "<end>":
        break
    if word not in ["<start>", "<pad>"]:
        caption_words.append(word)
caption_sentence = " ".join(caption_words)

# Print true and generated captions
true_captions = true_captions[0]
true_captions = '\n'.join(true_captions)
print('\nTrue captions:\n', true_captions)
print('\nGenerated caption:\n', caption_sentence)

plt.figure(figsize=(30, 50))
plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu())
plt.xticks([])
plt.yticks([])
plt.text(0, 240, 'Generated:', fontsize=35, color='b')
plt.text(0, 250, caption_sentence, fontsize=35)
plt.text(0, 280, 'True:', fontsize=35, color='g')
plt.text(0, 320, true_captions, fontsize=35)
plt.savefig('instance.png')
plt.close()

# Visualize attention
attention_weights = attention_weights.squeeze(0).squeeze(1).cpu().numpy()  # Shape: (seq_len, H*W)
seq_len, attn_size = attention_weights.shape  # (Number of words, H*W)
caption_words = caption_words[:seq_len]  # Truncate words to match attention sequence

# Get spatial dimensions of attention
H = W = int(np.sqrt(attn_size))  # Assuming attention is a square grid

# Normalize the image for display
image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)  # Normalize safely

# Create attention visualization
fig, axes = plt.subplots(1, len(caption_words) + 1, figsize=(20, 5))

# Show original image first
axes[0].imshow(image_np)
axes[0].set_title("Original Image", fontsize=12)
axes[0].axis("off")

# Show attention maps for each word
for t, word in enumerate(caption_words):
    attn_map = attention_weights[t].reshape(H, W)  # Reshape into grid

    # Normalize attention safely
    attn_map = attn_map - attn_map.min()  # Ensure no negative values
    if attn_map.max() > 0:  # Avoid divide-by-zero
        attn_map = attn_map / attn_map.max()

    # Resize attention to match image size
    attn_map = zoom(attn_map, (image_np.shape[0] / H, image_np.shape[1] / W), order=1)

    # Overlay attention heatmap
    axes[t + 1].imshow(image_np)  # Show original image
    axes[t + 1].imshow(attn_map, cmap="gray", alpha=0.7)  # Overlay attention heatmap
    axes[t + 1].set_title(word, fontsize=12)
    axes[t + 1].axis("off")

# Save the visualization
plt.tight_layout()
plt.savefig("instance_attention.png")
plt.close()