import argparse
import time
import os
import logging
import random
import pickle
import shutil
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from torchvision import transforms,utils
from model import Modified3DUNet
from celldataset import cell_training,IndexTracker,IndexTracker2,cell_testing,cell_testing_inter,cell_testing_PNAS
from utils import Parser, criterions
#import matplotlib
#import matplotlib.pyplot as plt
from skimage.transform import resize
import losses
from losses import Precision_img,Recall_img,F1_score_img
import time
from skimage.morphology import closing,binary_closing,binary_opening

parser = argparse.ArgumentParser()
parser.add_argument('-cfg','--cfg',default = 'cell',type=str)

path = os.path.dirname(__file__)

# parse arguments
args = parser.parse_args()
args = Parser(args.cfg, args)
ckpts = args.getdir()

def saveimage(image,filename):
    data = sitk.GetImageFromArray(image)
    sitk.WriteImage(data,filename)

def main():
    start_time = time.time()
    # setup environments and seeds
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # setup networks
    #Network = getattr(models, args.net)
    #model = Network(**args.net_params)

    model = Modified3DUNet(in_channels = 1,n_classes = 2, base_n_filter = 16)
    #load_model
    model_file = os.path.join(ckpts,'model_last.tar')
    print model_file
    checkpoint = torch.load(model_file,
            map_location = lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    '''optimizer = getattr(torch.optim, args.opt)(
            model.parameters(), **args.opt_params)'''
    #optimizer = torch.optim.SGD(model.parameters(),lr = 0.1,momentum=0.9)

    criterion = getattr(criterions, args.criterion)
    num_gpus = len(args.gpu.split(','))
    args.batch_size *= num_gpus
    args.workers    *= num_gpus

    # create dataloaders
    #Dataset = getattr(datasets, args.dataset)
    dset = cell_testing_inter('/home/tom/data1_match/dataset4/')
    print dset.__len__()
    test_loader = DataLoader(                                                                  
            dset, batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory = True)
    model.eval()
    torch.set_grad_enabled(False)
    inputs = []
    outputs=[]
    ground_truth = []
    for i, sample in enumerate(test_loader):
        input = sample['data']
        img_size = sample['image_size']
        print img_size[0]
        file_name = sample['name']
        file_name = str(file_name[0])
        _,_,z,x,y = input.shape
        seg = np.zeros((z,x,y))
        target = sample['seg']
        for j in range (z/16):       
            #ground_truth.append(target)
            #print file_name[0]
            input_temp = input[0,0,j*16:(j+1)*16].float()
            input_temp = input_temp[None,None,...]
            output_temp = nn.parallel.data_parallel(model, input_temp)
            output_temp = output_temp.detach().cpu().numpy()
            output_temp = output_temp[0]
            seg_temp = output_temp.argmax(0)
            seg[j*16:(j+1)*16] = seg_temp    
        data = input.detach().numpy()
        #print data.shape
        data = data[0,0,:,:,:]
        data = (255*data[0:5*img_size[0]]).astype('uint8')
        data_img = sitk.GetImageFromArray(data)
        sitk.WriteImage(data_img,'/home/tom/result/'+file_name)
        #outputs.append(output)
        #output = output[0]
        #print output.shape
        #seg = output.argmax(0)
        seg = (seg[0:5*img_size[0]]*255).astype('uint8')
        seg = seg.astype('float32')
        seg = seg/255.0
        seg = np.multiply(data,seg)
        result = np.zeros(img_size)
        result = seg[0:img_size[0],0:512,0:512]
        '''
        result = result/255
        threshold = 0.05
        result[result>0.06] = 1
        result[result<=0.02] = 0
        result = binary_closing(result)
        gt = target[0]
        gt = gt[0]
        gt = gt[0:img_size[0],0:512,0:512]
        gt = gt.numpy()
        print ("precision:%f",Precision_img(result,gt))
        print ("Recall:%f",Recall_img(result,gt))
        print ("f1_score:%f",F1_score_img(result,gt))'''
        result = (result*255).astype('uint8')
        seg = sitk.GetImageFromArray(result)
        sitk.WriteImage(seg,'/home/tom/membrane/'+file_name+'mem.tif')
        print ("running time %s"%(time.time()-start_time))
    #print len(inputs)
    #print len(outputs)
    #inputs = np.concatenate(inputs)
    #outputs = np.concatenate(outputs)
    #ground_truth = np.concatenate(ground_truth)
    #np.save('/home/tom/Modified-3D-UNet-Pytorch/output/inputs.npy',inputs)
    #np.save('/home/tom/Modified-3D-UNet-Pytorch/output/outputs.npy',outputs)
    #np.save('/home/tom/Modified-3D-UNet-Pytorch/output/ground.npy',ground_truth)


if __name__ == '__main__':
    main()
    #data = np.load('/home/tom/Modified-3D-UNet-Pytorch/output/inputs.npy')
    #seg = np.load('/home/tom/Modified-3D-UNet-Pytorch/output/outputs.npy')
    #ground_truth = np.load('/home/tom/Modified-3D-UNet-Pytorch/output/ground.npy')
    '''seg_1 =  seg[0]
    seg_1 = seg_1.argmax(0)
    print seg_1.shape
    exit(0)'''
    '''i = 0
    for image in data:
    #precision = losses.Precision(seg,ground_truth)
    #recall = losses.Recall(seg,ground_truth)
    #F1 = losses.F1_score(seg,ground_truth)
    #print precision,recall,F1
        data = data[0]
        data = data[0,:,:,:]
        data = (data*255).astype('uint8')
        seg_1 = seg[0]
        seg = seg_1.argmax(0)
        seg = (seg*255).astype('uint8')
        seg = sitk.GetImageFromArray(seg)
        sitk.WriteImage(seg,'/home/tom/result1.tif')
        data = sitk.GetImageFromArray(data)
        sitk.WriteImage(data,'/home/tom/data.tif')

    fig,ax = plt.subplots(1,1)
    tracker = IndexTracker(ax,seg)
    fig.canvas.mpl_connect('scroll_event',tracker.onscroll)
    plt.show()'''


