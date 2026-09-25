import torch
from mine_att import EncoderCNN, DecoderRNN, Vocabulary
from pycocotools.coco import COCO
from torchvision import transforms, models
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import pickle
import numpy as np
from PIL import Image

from mine_att import GroupedTestDataset

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

# ImageNet transform for model input
transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()

# Load test dataset
test_dataset = torch.load("test_dataset.pth", weights_only=False)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    collate_fn=lambda x: (torch.stack([xi[0] for xi in x]), [xi[1] for xi in x], [xi[2] for xi in x])
)

from mine_att import encoder_linear_dropout_rate, embed_size, hidden_size, embed_dropout_rate, lstm_dropout_rate, num_layers

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

# Get original image path from COCO and load it
img_info = full_coco.loadImgs(img_id[0])[0]
original_image = Image.open(f'/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/train2017/{img_info["file_name"]}').convert('RGB')
original_image_np = np.array(original_image) / 255.0  # Convert to [0,1] range

# Prepare image for model
image = transform(original_image).unsqueeze(0).to(device)

# Generate caption and attention weights
with torch.no_grad():
    features = encoder(image)
    captions, attention_weights = decoder.generate_beam_with_attention(features, beam_size=7, max_length=30)

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

# Save instance image with caption
plt.figure(figsize=(30, 50))
plt.imshow(original_image_np)
plt.xticks([])
plt.yticks([])
plt.text(0, original_image_np.shape[0] + 40, 'Generated:', fontsize=35, color='b')
plt.text(0, original_image_np.shape[0] + 80, caption_sentence, fontsize=35)
plt.text(0, original_image_np.shape[0] + 120, 'True:', fontsize=35, color='g')
plt.text(0, original_image_np.shape[0] + 200, true_captions, fontsize=35)
plt.savefig('instance.png', bbox_inches='tight')
plt.close()

# Visualize attention on original image
attention_weights = attention_weights.squeeze(0).squeeze(1).cpu().numpy()
seq_len, attn_size = attention_weights.shape
caption_words = caption_words[:seq_len]

H = W = int(np.sqrt(attn_size))  # Attention grid size from model

# Create figure with appropriate number of subplots
fig, axes = plt.subplots(1, len(caption_words) + 1, figsize=(20, 5))

# Show original image first
axes[0].imshow(original_image_np)
axes[0].set_title("Original Image", fontsize=12)
axes[0].axis("off")

# Calculate scaling factors
scale_h = original_image_np.shape[0] / H
scale_w = original_image_np.shape[1] / W

# Visualize attention for each word
for t, word in enumerate(caption_words):
    attn_map = attention_weights[t].reshape(H, W)

    # Normalize attention
    attn_map = attn_map - attn_map.min()
    if attn_map.max() > 0:
        attn_map = attn_map / attn_map.max()

    # Resize attention map
    attn_map = zoom(attn_map, (scale_h, scale_w), order=1)

    # Overlay on original image
    axes[t+1].imshow(original_image_np)
    axes[t+1].imshow(attn_map, cmap="grey", alpha=0.8)
    axes[t+1].set_title(word, fontsize=12)
    axes[t+1].axis("off")

plt.tight_layout()
plt.savefig("instance_attention.png", bbox_inches='tight')
plt.close()
