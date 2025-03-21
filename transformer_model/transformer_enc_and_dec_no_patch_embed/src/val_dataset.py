
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import os

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