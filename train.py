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
from dataset import build_dataset
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

def train():
    # enable mixed-precision computation if desired
    if args.amp:
        mixed_precision.enable_mixed_precision()
    
    stl_batch_size = 400    
    train_loader, test_loader, stl_train_loader, stl_test_loader = build_dataset(args.batch_size, stl_batch_size)

    caption_encoder = CaptionEncoder(args.n_rkhs, args.cap_seq_len, hidden_size=args.cap_fc_size, device=device)
    resnet50 = ResNet50Encoder(encoder_size=128, n_rkhs=args.n_rkhs, ndf=args.ndf, n_depth=args.n_depth)
    resnet50.init_weights()
    resnet50 = resnet50.to(device)

    #stl10_fc = nn.Linear(in_features=args.n_rkhs, out_features=10)
    #stl10_fc = stl10_fc.to(device)

    optimizer = torch.optim.Adam(
        [{'params': mod.parameters(), 'lr': args.learning_rate} for mod in [caption_encoder.conv, resnet50]],
        betas=(0.8, 0.999), weight_decay=1e-5, eps=1e-8)
    nce = LossMultiNCE().to(device)

    # lin_optimizer = torch.optim.Adam([
    #     {'params': stl10_fc.parameters()}
    # ])

    if args.cpt_load_path is not None:
        ckpt = torch.load(args.cpt_load_path)
        resnet50.load_state_dict(ckpt['resnet50'])
        caption_encoder.conv.load_state_dict(ckpt['caption_conv'])
        print('Checkpoint loaded.')

    resnet50, optimizer = mixed_precision.initialize(resnet50, optimizer)
    #stl10_fc, lin_optimizer = mixed_precision.initialize(stl10_fc, lin_optimizer)
    #fc_loss = nn.CrossEntropyLoss()
    
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
            #loss = loss_1t1 + lgt_reg
            optimizer.zero_grad()
            mixed_precision.backward(loss, optimizer)
            #loss.backward()
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
            if step % 100 == 0:
                print('Time per step: ', (time.time() - t0) / 100.0)
                t0 = time.time()
            step += 1
        # Test:
        # For each image, retrieve top 5 captions and check if the actual one is in there
        # For each caption, retrieve top 5 images and check if actual one is in there

        correct = 0
        total = 0
        for _, ((raw_imgs, images), captions) in enumerate(test_loader):
            images = images.to(device)
            captions = captions[random.randint(0, 4)]
            r1, r7 = resnet50(images)
            encoded_captions, word_reps = caption_encoder(captions)
            encoded_captions = encoded_captions.to(device)
            # (batch_size, 5)
            cos_sims_idx = nce_retrieval(r1, encoded_captions)
            y = torch.arange(0, images.size(0))
            correct += (cos_sims_idx.cpu().t() == y).sum().item()    
            total += images.size(0)
        print('Epoch {}, test top-5 retrieval accuracy: {}'.format(epoch, correct / total))
        wandb.log({'test_top_5': correct / total})

        # if epoch % 3 == 0:
        #     correct_count = 0
        #     total = 0
        #     for fc_epoch in range(15):
        #         for images, labels in stl_train_loader:
        #             images = images.to(device)
        #             labels = labels.to(device)
        #             r1 = resnet50(images).reshape(stl_batch_size, args.n_rkhs)
        #             r1 = r1.detach()  # we don't want the labels to impact the encoder
        #             out = stl10_fc(r1)
        #             class_loss = fc_loss(out, labels)
        #             wandb.log({'cls loss': class_loss})
        #             lin_optimizer.zero_grad()
        #             class_loss.backward()
        #             lin_optimizer.step()
        #     for images, labels in stl_test_loader:
        #         images = images.to(device)
        #         labels = labels.to(device)
        #         r1 = resnet50(images).reshape(stl_batch_size, args.n_rkhs)
        #         out = stl10_fc(r1)
        #         correct_count += get_correct_count(out.cpu(), labels.cpu())
        #         total += labels.size(0)
        #     print('Test Accuracy:', correct_count / total)
        #     wandb.log({'STL Accuracy': correct_count / total})

        os.makedirs('./checkpoints', exist_ok=True)
        torch.save({    
            'epoch': epoch,
            'resnet50': resnet50.state_dict(),
            'caption_conv': caption_encoder.conv.state_dict(),
            #'stl_fc': stl10_fc.state_dict(),
            'optimizer': optimizer.state_dict(),
            #'lin_optimizer': lin_optimizer.state_dict()
        }, 'checkpoints/{}_model.pth'.format(run_id))
            

if __name__ == '__main__':
    train()
