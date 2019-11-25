import os
import argparse
from enum import Enum
import random 

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid

import wandb

from model import ResNet50Encoder
from dataset import build_dataset
from nce import LossMultiNCE, nce_retrieval
from caption_encoder import CaptionEncoder

SEQ_LEN = 20
N_RKHS = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=1)
args = parser.parse_args()
run_id = wandb.init(project='amdim-bert').id

def get_correct_count(lgt_vals, lab_vals):
    # count how many predictions match the target labels
    max_lgt = torch.max(lgt_vals.cpu().data, 1)[1]
    num_correct = (max_lgt == lab_vals).sum().item()
    return num_correct


def vizualize(raw_images, encoded_images, raw_queries, encoded_queries):
    # Reuse the already encoded images from the batch
    top_k = 5
    top_k_idx = nce_retrieval(encoded_images, encoded_queries, top_k)
    top_k_idx = torch.flatten(top_k_idx)
    matches = raw_images[top_k_idx]
    viz = make_grid(matches, nrow=top_k)
    wandb.log({'viz': wandb.Image(viz)})
    wandb.log({'queries': wandb.Table(data=raw_queries, columns=['Query'])})

def train():
    stl_batch_size = 400
    train_loader, test_loader, stl_train_loader, stl_test_loader = build_dataset(args.b, stl_batch_size)

    caption_encoder = CaptionEncoder(N_RKHS, SEQ_LEN, device)
    # bert.to(device)
    resnet50 = ResNet50Encoder(encoder_size=128, n_rkhs=N_RKHS)
    resnet50.init_weights()
    resnet50.to(device)

    fc = nn.Linear(in_features=N_RKHS, out_features=10)
    fc = fc.to(device)

    optimizer = torch.optim.Adam(
        [{'params': mod.parameters(), 'lr': 0.0001} for mod in [caption_encoder.fc, resnet50]],
        betas=(0.8, 0.999), weight_decay=1e-5, eps=1e-8)
    nce = LossMultiNCE().to(device)

    lin_optimizer = torch.optim.Adam([
        {'params': fc.parameters()}
    ])

    fc_loss = nn.CrossEntropyLoss()
    
    train_encoder = True
    for epoch in range(500):
        print('epoch %i...' % epoch)
        step = 0
        if train_encoder:
            for _, ((raw_imgs, images), captions) in enumerate(train_loader):
                images = images.to(device)
                encoded_images = resnet50(images)

                # Each images has 5 captions, randomly select one of them
                captions = captions[random.randint(0,4)]             
                encoded_captions = caption_encoder(captions)

                loss, reg = nce(encoded_images, encoded_captions)
                loss = loss + reg
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if step % 10 == 0:
                    wandb.log({'loss': loss})
                    test_queries = encoded_captions[:3]
                    vizualize(raw_imgs, encoded_images, captions[:3], test_queries)
                step += 1

        if epoch % 2 == 0 or epoch < 3:
            correct_count = 0
            total = 0
            for fc_epoch in range(15):
                for images, labels in stl_train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    r1 = resnet50(images).reshape(stl_batch_size, N_RKHS)
                    r1 = r1.detach()  # we don't want the labels to impact the encoder
                    out = fc(r1)
                    class_loss = fc_loss(out, labels)
                    wandb.log({'cls loss': class_loss})
                    lin_optimizer.zero_grad()
                    class_loss.backward()
                    lin_optimizer.step()
            for images, labels in stl_test_loader:
                images = images.to(device)
                labels = labels.to(device)
                r1 = resnet50(images).reshape(stl_batch_size, N_RKHS)
                out = fc(r1)
                correct_count += get_correct_count(out.cpu(), labels.cpu())
                total += labels.size(0)
            print('Test Accuracy:', correct_count / total)
            wandb.log({'STL Accuracy': correct_count / total})

        torch.save({    
            'epoch': epoch,
            'resnet50': resnet50.state_dict(),
            'caption_fc': caption_encoder.fc.state_dict(),
            'stl_fc': fc.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lin_optimizer': lin_optimizer.state_dict()
        }, '{}_model.pth'.format(run_id))
            

if __name__ == '__main__':
    train()