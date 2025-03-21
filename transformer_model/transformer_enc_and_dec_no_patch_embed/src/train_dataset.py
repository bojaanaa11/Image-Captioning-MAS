from pycocotools.coco import COCO
from PIL import Image
import torch
import os
from torch.utils.data import Dataset

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