from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import os
from torchvision import models

transform = models.ResNet50_Weights.IMAGENET1K_V2.transforms()

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