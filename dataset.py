import os

from torchvision import datasets, transforms
import torch

INTERP = 3

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
        self.raw_trans = transforms.Compose([
            transforms.Resize(256, interpolation=INTERP),
            transforms.CenterCrop(256),
            transforms.ToTensor()
        ])
    
    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        orig = self.raw_trans(inp)
        return orig, out1


def get_dataset(dataset: str, batch_size: int):
    transforms128 = Transforms128()
    if dataset == 'coco':
        train_dataset = datasets.CocoCaptions(
                                        root=os.path.expanduser('~/data/coco/train2017'), 
                                        annFile=os.path.expanduser('~/data/coco/annotations/captions_train2017.json'), 
                                        transform=transforms128)
        
        print('WARNING: test dataset is using train transforms')
        test_dataset = datasets.CocoCaptions(
                                        root=os.path.expanduser('~/data/coco/val2017'), 
                                        annFile=os.path.expanduser('~/data/coco/annotations/captions_val2017.json'), 
                                        transform=transforms128)
    elif dataset == 'stl10':
        train_dataset = datasets.STL10(root=os.path.expanduser('~/data'), transform=transforms128.test_transform)
        test_dataset = datasets.STL10(root=os.path.expanduser('~/data'), transform=transforms128.test_transform, split='test')


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
