import json
import os
import argparse
from enum import Enum

from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from transformers import DistilBertTokenizer, DistilBertModel

from model import ResNet50Encoder
from nce import LossMultiNCE

INTERP = 3

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=1)
args = parser.parse_args()


class Transforms128:
    '''
    ImageNet dataset, for use with 128x128 full image encoder.
    '''
    def __init__(self):
        # image augmentation functions
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        rand_crop = \
            transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=INTERP)
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize(146, interpolation=INTERP),
            transforms.CenterCrop(128),
            post_transform
        ])
        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            post_transform
        ])


def build_dataset(batch_size: int):
    transforms128 = Transforms128()
    train_dataset = datasets.CocoCaptions(
                                    root=os.path.expanduser('~/data/coco/train2017'), 
                                    annFile=os.path.expanduser('~/data/coco/annotations/captions_train2017.json'), 
                                    transform=transforms128.train_transform)
    
    test_dataset = datasets.CocoCaptions(
                                    root=os.path.expanduser('~/data/coco/val2017'), 
                                    annFile=os.path.expanduser('~/data/coco/annotations/captions_val2017.json'), 
                                    transform=transforms128.test_transform)

    # build pytorch dataloaders for the datasets
    train_loader = \
        torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True,
                                    num_workers=16)
    test_loader = \
        torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True,
                                    num_workers=16)

    return train_loader, test_loader

def build_encoded_ann_file():
    batch_size = 10
    train_loader, test_loader = build_dataset(batch_size)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

    i = 0
    enc_buf = []
    for _, annotations in train_loader:
        anns = [a[0] for a in annotations]
        encodings = [torch.tensor(tokenizer.encode(s)) for s in anns]
        seq_len = 50
        padded = torch.stack(
            [torch.cat([e, torch.zeros((seq_len - len(e)), dtype=torch.long)]) if len(e) < seq_len 
             else e[:seq_len]
             for e in encodings])
        out = bert(padded)
        # for e in out:
        #     enc_buf.append(e.tolist())
        # Encode every annotation and save it somewhere
        # Compare representation of padded vs non-padded sentence, should be about the same
        # Do crossproduct of related vs unrelated sentences, related should have a higher score
        i += batch_size
        if i % 1000 == 0:
            print(i)
            # with open('enc_anns_train.json', 'a+') as f:
            #     json.dump(enc_buf, f)
            # enc_buf = []
    print(i)

if __name__ == "__main__":
    build_encoded_ann_file()