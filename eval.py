from models.models import *
from datasets import HDF5Dataset
from deploy import test
from utils import load_checkpoint

from torch.utils.data import DataLoader
from transforms.transforms import microdoppler_transform

import os
import functools
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


parser = argparse.ArgumentParser(description='Indoor Person Identification')
parser.add_argument('--params', default='./params/mymodel_bvalid.pt', type=str)
parser.add_argument('--name', default='dbg', type=str)
parser.add_argument('--network', default='DSDA', type=str)
parser.add_argument('--dataset', default='/mnt/hdd/dbstjswo505/workspace/MDPI_Sensor/IDRad1_Dataset/idrad', type=str)
parser.add_argument('--targets', default=['target1', 'target2', 'target3', 'target4', 'target5'], nargs='+', type=str) # ,
parser.add_argument('--features', default='microdoppler_thresholded', type=str)
parser.add_argument('--learning_rate', default=10 ** -3, type=float)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--max_epochs', default=1, type=int)
parser.add_argument('--test', action='store_true')
parser.set_defaults(test=False)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Initialize network
net = eval(args.network)(input_dim=(1, 45, 205), output_dim=len(args.targets))
net = net.cuda()

optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=0.0005)
criterion = nn.CrossEntropyLoss()

if args.params != "":
    load_checkpoint(net, None, args.params)

values = dict()
values['microdoppler'] = {'mean':-43898.7272684, 'std':1457.87896807, 'min':-46583.265625, 'max':-29791.0058594}
values['microdoppler_thresholded'] = {'mean':-16987.4060019, 'std':619.551479691, 'min':-17100.0, 'max':-6727.47998047}

transform = functools.partial(microdoppler_transform, values=values[args.features], standard_scaling=True, preprocessing=True)

dataset = dict(train=HDF5Dataset(os.path.join(args.dataset, 'train'), args.targets, args.features, transform=transform, in_memory=True, random_shift=True),
               valid=HDF5Dataset(os.path.join(args.dataset, 'valid'), args.targets, args.features, transform=transform, in_memory=True),
               test=HDF5Dataset(os.path.join(args.dataset, 'test'), args.targets, args.features, transform=transform, in_memory=True))

train_loader = DataLoader(dataset["train"], batch_size=args.batch_size, num_workers=12, shuffle=True)
valid_loader = DataLoader(dataset["valid"], batch_size=args.batch_size, num_workers=12)
test_loader = DataLoader(dataset["test"], batch_size=args.batch_size, num_workers=12)

print("---------")
print("%d samples and %d batches in train set." % (len(dataset['train']), len(train_loader)))
print("%d samples and %d batches in validation set." % (len(dataset['valid']), len(valid_loader)))
print("%d samples and %d batches in test set." % (len(dataset['test']), len(test_loader)))
print("---------")

if not args.test:
    test(net,
          dict(train=train_loader, valid=valid_loader, test=test_loader),
          args.name,
          optimizer=optimizer,
          criterion=criterion,
          max_epochs=args.max_epochs,
          phases=['train','valid','test'],
          classlabels=args.targets)
