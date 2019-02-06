'''import argparse
import os
import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import load_dataset, BraTS2018List
from model import Modified3DUNet
import paths

def datestr():
	now = time.localtime()
	return '{:04}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

#print datestr()



# Training setting
parser = argparse.ArgumentParser(description='PyTorch Modified 3D U-Net Training')
#parser.add_argument('-m', '--modality', default='T1', choices = ['T1', 'T1c', 'T2', 'FLAIR'],
#                    type = str, help='modality of input 3d images (default:T1)')
#parser.add_argument('-w', '--workers', default=8, type=int,
#                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run (default: 300)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    help='batch size (default: 2)')
parser.add_argument('-g', '--gpu', default='0', type=str)
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    help='initial learning rate (default:5e-4)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=985e-3, type=float,
                     help='weight decay (default: 985e-3)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 100)')
parser.add_argument('-d', '--data', default=paths.preprocessed_training_data_folder,
                    type=str, help='The location of BRATS2015')

log_file = os.path.join("train_log.txt")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=log_file)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger('').addHandler(console)


global args, best_loss
best_loss = float('inf')
args = parser.parse_args()
#print os.environ['CUDA_VISIBLE_DEVICES']
dtype = torch.float
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# input = data.to(device)

# Loading the model
in_channels = 4
n_classes = 4
base_n_filter = 16
model = Modified3DUNet(in_channels, n_classes, base_n_filter).to(device)
#print args.data


# Split the training and testing dataset

test_size = 0.1
train_idx, test_idx = train_test_split(range(285), test_size = test_size)
train_data = load_dataset(train_idx)
test_data = load_dataset(test_idx)


#print all_data.keys()
# create your optimizer
#optimizer = optim.adam(net.parameteres(), lr=)

# in training loop:
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step
'''
import argparse
import time
import os
import logging
import random
import pickle
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from torchvision import transforms,utils
from model import Modified3DUNet
from celldataset import cell_training,cell_training_patch
from utils import Parser, criterions


parser = argparse.ArgumentParser()
parser.add_argument('-cfg','--cfg',default = 'cell',type=str)

path = os.path.dirname(__file__)

# parse arguments
args = parser.parse_args()
args = Parser(args.cfg, args)
ckpts = args.makedir()

# setup logs
log = os.path.join(path,'logs',args.cfg+'.txt')
fmt = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt, filename=log)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter(fmt))
logging.getLogger('').addHandler(console)


def main():
    # setup environments and seeds
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # setup networks
    #Network = getattr(models, args.net)
    #model = Network(**args.net_params)

    model = Modified3DUNet(in_channels = 1,n_classes = 2, base_n_filter = 16)
    model = model.cuda()
    '''optimizer = getattr(torch.optim, args.opt)(
            model.parameters(), **args.opt_params)'''
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001,weight_decay =0.0001)
    #optimizer = torch.optim.SGD(model.parameters(),lr = 0.1,momentum=0.9)

    criterion = getattr(criterions, args.criterion)
    msg = '-------------- New training session -----------------'
    msg += '\n' + str(args)
    logging.info(msg)
    num_gpus = len(args.gpu.split(','))
    args.batch_size *= num_gpus
    args.workers    *= num_gpus
    args.opt_params['lr'] *= num_gpus
    # create dataloaders
    #Dataset = getattr(datasets, args.dataset)
    dset = cell_training('/home/tom/Modified-3D-UNet-Pytorch/PNAS/')
    train_loader = DataLoader(                                                                  
            dset, batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True)
    file_name_best = os.path.join(ckpts, 'cell/model_best.tar')                              
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train_loss = train(
                train_loader, model, criterion, optimizer, epoch)
        # remember best lost and save checkpoint                                    
        ckpt = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            }                                                                               
        file_name = os.path.join(ckpts, 'model_last.tar')
        torch.save(ckpt, file_name)
        msg = 'Epoch: {:02d} Train loss {:.4f}'.format(
                epoch+1, train_loss)
        logging.info(msg)

def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    # switch to train mode
    model.train()
    torch.set_grad_enabled(True) # enable_grad

    start = time.time()

    for i, sample in enumerate(train_loader):
        input = sample['data']
        #print input.size()
        target = sample['seg']
        target = target.cuda(non_blocking=True)

        # compute output
        output = nn.parallel.data_parallel(model, input)
        #output = model(input)
        loss   = criterion(output, target)
        # measure accuracy and record loss
        #losses.update(loss.item(), input.size(0))
        losses.update(loss.item(), 1)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        #a = list(model.parameters())[0].clone()
        loss.backward()
        optimizer.step()
        #b = list(model.parameters())[0].clone()
        #print ('parameters change?',torch.equal(a.data,b.data))
    return losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n  
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    # reduce learning rate by a factor of 10
    if epoch+1 in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


if __name__ == '__main__':
    main()


