
from pycocotools.coco import COCO

import torch
from torch.utils.data import DataLoader

import numpy as np
import pickle

from src.vocabulary import Vocabulary
from src.train_dataset import CocoDataset
from src.val_dataset import CocoValidationDataset, CocoValidationLossDataset
from src.test_dataset import GroupedTestDataset
import src.parameters as params

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    captions = [item[1] for item in batch]
    img_ids = [item[2] for item in batch]
    return images, captions, img_ids

def prepare():
    # Prepare
    # Load full training dataset
    full_coco = COCO('/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/annotations/captions_train2017.json')
    all_img_ids = np.array(full_coco.getImgIds())

    # Split 80-20 (bc there is no testing data so taking it from train)
    np.random.seed(42)
    np.random.shuffle(all_img_ids)
    split_idx = int(0.8 * len(all_img_ids))
    train_img_ids = all_img_ids[:split_idx]
    test_img_ids = all_img_ids[split_idx:]

    print(f'Number of all images: {len(all_img_ids)}')
    print(f'Number of train images: {len(train_img_ids)}')
    print(f'Number of test images: {len(test_img_ids)}')

    # Build vocabulary using ONLY training captions
    train_ann_ids = full_coco.getAnnIds(imgIds=train_img_ids)
    train_captions = [full_coco.anns[ann_id]['caption'] for ann_id in train_ann_ids]

    vocab = Vocabulary(params.freq_threshold)
    vocab.build_vocabulary(train_captions)
    with open("/home/obojana/bojana/src_saved/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print(f'Vocab length: {len(vocab)} \n')

    # Train
    # Training Dataset (80%)
    train_dataset = CocoDataset(
        root='/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/train2017/',
        annFile='/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/annotations/captions_train2017.json',
        vocab=vocab,
        transform=params.transform,
        max_length=params.max_length,
        img_ids=train_img_ids.tolist()
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=params.batch_size,
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
            transform=params.transform,
            max_length=params.max_length
        ),
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=4
    )

    # For validation METRICS (grouped by image)
    val_metrics_loader = DataLoader(
        CocoValidationDataset(
            root='/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/val2017/',
            annFile='/home/obojana/.cache/kagglehub/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017/annotations/captions_val2017.json',
            transform=params.transform
        ),
        batch_size=params.batch_size,
        collate_fn=collate_fn
    )

    # Test
    test_dataset = GroupedTestDataset(test_img_ids.tolist())
    test_loader = DataLoader(
        test_dataset,
        batch_size=params.batch_size,
        collate_fn=lambda x: (torch.stack([xi[0] for xi in x]), [xi[1] for xi in x], [xi[2] for xi in x])
    )

    # Saving for later use
    torch.save(test_loader.dataset, "/home/obojana/bojana/src_saved/test_dataset.pth")

    return train_loader, val_loss_loader, val_metrics_loader, test_loader, vocab