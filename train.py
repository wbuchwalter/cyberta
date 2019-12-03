import os
import argparse
from enum import Enum
import random 
import time

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid

import wandb

import mixed_precision
from model import ResNet50Encoder
from dataset import get_dataset
from nce import LossMultiNCE, nce_retrieval
from caption_encoder import CaptionEncoder

parser = argparse.ArgumentParser()
parser.add_argument('--cap_fc_size', type=int, default=4096)
parser.add_argument('--cap_seq_len', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=200,
                    help='input batch size (default: 200)')
parser.add_argument('--learning_rate', type=float, default=0.0002,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Enables automatic mixed precision')

# parameters for model and training objective
parser.add_argument('--ndf', type=int, default=128,
                    help='feature width for encoder')
parser.add_argument('--n_rkhs', type=int, default=1024,
                    help='number of dimensions in fake RKHS embeddings')
parser.add_argument('--tclip', type=float, default=20.0,
                    help='soft clipping range for NCE scores')
parser.add_argument('--n_depth', type=int, default=3)
parser.add_argument('--use_bn', type=int, default=0)

# parameters for output, logging, checkpointing, etc
parser.add_argument('--cpt_load_path', type=str, default=None,
                    help='path from which to load checkpoint (if available)')
args = parser.parse_args()

run_id = wandb.init(project='amdim-bert', config=args).id
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


def test_coco_retrieval(test_loader, resnet, caption_encoder):
    correct = 0
    total = 0
    for _, ((_, images), captions) in enumerate(test_loader):
        images = images.to(device)
        captions = captions[random.randint(0, 4)]
        r1, _ = resnet50(images)
        encoded_captions, _ = caption_encoder(captions)
        encoded_captions = encoded_captions.to(device)
        # (batch_size, 5)
        cos_sims_idx = nce_retrieval(r1, encoded_captions)
        y = torch.arange(0, images.size(0))
        correct += (cos_sims_idx.cpu().t() == y).sum().item()    
        total += images.size(0)
    return correct / total
    
def train():
    # enable mixed-precision computation if desired
    if args.amp:
        mixed_precision.enable_mixed_precision()
    
    train_loader, test_loader = get_dataset('coco', args.batch_size)
    caption_encoder = CaptionEncoder(args.n_rkhs, args.cap_seq_len, hidden_size=args.cap_fc_size, device=device)
    resnet50 = ResNet50Encoder(n_rkhs=args.n_rkhs, ndf=args.ndf, n_depth=args.n_depth)
    resnet50 = resnet50.to(device)

    optimizer = torch.optim.Adam(
        [{'params': mod.parameters(), 'lr': args.learning_rate} for mod in [caption_encoder.conv, resnet50]],
        betas=(0.8, 0.999), weight_decay=1e-5, eps=1e-8)
    nce = LossMultiNCE().to(device)

    if args.cpt_load_path is not None:
        ckpt = torch.load(args.cpt_load_path)
        resnet50.load_state_dict(ckpt['resnet50'])
        caption_encoder.conv.load_state_dict(ckpt['caption_conv'])
        print('Checkpoint loaded.')

    resnet50, optimizer = mixed_precision.initialize(resnet50, optimizer)

    top_5_accuracy = test_coco_retrieval(test_loader, resnet50, caption_encoder)
    print('Before training, test top-5 retrieval accuracy: {}'.format(epoch,top_5_accuracy))    

    for epoch in range(500):
        print('epoch %i...' % epoch)
        step = 0
        t0 = time.time()
        for _, ((raw_imgs, images), captions) in enumerate(train_loader):
            images = images.to(device)
            # Each images has 5 captions, randomly select one of them
            captions = captions[random.randint(0, 4)]

            r1, r7 = resnet50(images)
            encoded_captions, word_reps = caption_encoder(captions)
            encoded_captions = encoded_captions.to(device)

            loss_gtg, loss_gtl, loss_ltl, lgt_reg = nce(r1, r7, encoded_captions, word_reps)
            loss = loss_gtg + loss_gtl + loss_ltl + lgt_reg
            optimizer.zero_grad()
            mixed_precision.backward(loss, optimizer)
            optimizer.step()
            if step % 50 == 0:
                wandb.log({
                    'loss': loss,
                    'loss_gtg': loss_gtg,
                    'loss_gtl': loss_gtl,
                    'loss_ltl': loss_ltl
                })

                # test_queries = encoded_captions[:3]
                # vizualize(raw_imgs, encoded_images, captions[:3], test_queries)
            if step % 100 == 99:
                print('Time per step: ', (time.time() - t0) / 100.0)
                t0 = time.time()
            step += 1

        top_5_accuracy = test_coco_retrieval(test_loader, resnet50, caption_encoder)
        print('Epoch {}, test top-5 retrieval accuracy: {}'.format(epoch,top_5_accuracy))
        wandb.log({'test_top_5': top_5_accuracy})

        if epoch % 5 == 0:
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save({    
                'epoch': epoch,
                'resnet50': resnet50.state_dict(),
                'caption_conv': caption_encoder.conv.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, 'checkpoints/{}_model.pth'.format(run_id))
            

if __name__ == '__main__':
    train()
