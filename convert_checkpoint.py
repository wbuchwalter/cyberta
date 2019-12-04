import argparse

import torch
import torch.nn as nn

from model import Encoder
from caption_encoder import CaptionEncoder

parser = argparse.ArgumentParser()
parser.add_argument('-c', type=str, required=True)

args = parser.parse_args()

class LoadingModule(nn.Module):
    def __init__(self, n_rkhs, ndf, n_depth):
        super(LoadingModule, self).__init__()
        self.encoder = Encoder(n_rkhs=n_rkhs, ndf=ndf, n_depth=n_depth)
        self.encoder = nn.DataParallel(self.encoder)


def convert(ckp_path):
    ckp = torch.load(ckp_path)
    hp = ckp['hyperparams']
    loading_mod = LoadingModule(n_rkhs=hp['n_rkhs'], ndf=hp['ndf'], n_depth=hp['n_depth'])
    #loading_mod.to(device)
    model_dict = loading_mod.state_dict()
    partial_params = {k: v for k, v in ckp['model'].items() if not (k.startswith("evaluator.") or k == "g2l_loss.masks_r5")}
    model_dict.update(partial_params)
    loading_mod.load_state_dict(model_dict)

    resnet50 = loading_mod.encoder.module
    caption_encoder = CaptionEncoder(n_rkhs=hp['n_rkhs'], seq_len=25, hidden_size=4096)
    torch.save({
        'epoch': 0,
        'resnet50': resnet50.state_dict(),
        'caption_conv': caption_encoder.conv.state_dict()
    }, 'converted_checkpoint.pth')

if __name__ == '__main__':
    convert(args.c)
