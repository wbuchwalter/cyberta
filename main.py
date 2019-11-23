import os
import argparse
from enum import Enum

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from transformers import DistilBertTokenizer, DistilBertModel
import wandb

from model import ResNet50Encoder
from nce import LossMultiNCE

INTERP = 3


wandb.init(project='amdim-bert')

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
    
    stl_train_dataset = datasets.STL10(root=os.path.expanduser('~/data'), transform=transforms128.test_transform)

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
    
    stl_train_loader = \
        torch.utils.data.DataLoader(dataset=stl_train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True,
                                    num_workers=16)

    return train_loader, test_loader, stl_train_loader


class CaptionEncoder(nn.Module):
    def __init__(self, n_rkhs, seq_len, device):
        super(CaptionEncoder, self).__init__()
        self.seq_len = seq_len
        self.device = device
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=seq_len * 768, out_features=seq_len * 768 // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=seq_len * 768 // 2, out_features=n_rkhs)
        )
        self.fc.to(device)
    
    def forward(self, x, attention_mask):
        batch_size = x.size(0)
        out = self.bert(x, attention_mask=attention_mask)[0]
        out = out.reshape(batch_size, 768 * self.seq_len)
        out = out.to(self.device)
        out = self.fc(out)
        return out


def get_correct_count(lgt_vals, lab_vals):
    # count how many predictions match the target labels
    max_lgt = torch.max(lgt_vals.cpu().data, 1)[1]
    num_correct = (max_lgt == lab_vals).sum().item()
    return num_correct

def train():
    n_rkhs = 512
    seq_len = 50

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: %s' % device)
    train_loader, test_loader, stl_train_loader = build_dataset(args.b)


    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    caption_encoder = CaptionEncoder(n_rkhs, seq_len, device)

    # bert.to(device)
    resnet50 = ResNet50Encoder(encoder_size=128, n_rkhs=n_rkhs)
    resnet50.init_weights()
    resnet50.to(device)

    fc = nn.Linear(in_features=n_rkhs, out_features=10)
    fc = fc.to(device)

    optimizer = torch.optim.Adam(
        [{'params': mod.parameters()} for mod in [caption_encoder, resnet50]],
        betas=(0.8, 0.999), weight_decay=1e-5, eps=1e-8)
    nce = LossMultiNCE().to(device)

    #lin_optimizer = torch.optim.Adam(fc.parameters())
    lin_optimizer = torch.optim.Adam([
        #{'params': resnet50.parameters()},
        {'params': fc.parameters()}
    ])

    fc_loss = nn.CrossEntropyLoss()
    
    for epoch in range(50):
        print('epoch %i...' % epoch)
        step = 0
        for images, annotations in test_loader:
            images = images.to(device)
            # annotations = annotations.to(device)
            anns = annotations[0]  # take the first caption for each image, this could be randomly selected later
            encodings = [torch.tensor(tokenizer.encode(s)) for s in anns]
            padded = torch.stack(
                [torch.cat([e, torch.zeros((seq_len - len(e)), dtype=torch.long)]) if len(e) < seq_len
                else e[:seq_len]
                for e in encodings])
            attn_mask = (padded > 0)
            #encoded_captions = bert(padded, attention_mask=attn_mask)[0]
            encoded_images = resnet50(images)
            encoded_captions = caption_encoder(padded, attn_mask)
            loss, reg = nce(encoded_images, encoded_captions)
            loss = loss + reg
            optimizer.zero_grad()
            loss.backward()
            # print(loss)
            # if step % 10 == 0:
            #     wandb.log({'loss': loss})
            optimizer.step()
            if step > 1:
                break
            step += 1

        if epoch % 2 == 0 or epoch < 3:
            correct_count = 0
            total = 0
            for fc_epoch in range(50):
                for images, labels in stl_train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    r1 = resnet50(images).reshape(args.b, n_rkhs)
                    #print('shapy', r1.shape)
                    out = fc(r1)
                    class_loss = fc_loss(out, labels)
                    wandb.log({'cls loss': class_loss})
                    lin_optimizer.zero_grad()
                    class_loss.backward()
                    lin_optimizer.step()
                    correct_count += get_correct_count(out.cpu(), labels.cpu())
                    total += labels.size(0)
                print('Accuracy:', correct_count / total)
            wandb.log({'STL Accuracy': correct_count / total})
            

if __name__ == '__main__':
    train()