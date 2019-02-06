import glob
import os
import numpy as np
from matplotlib import pyplot as plt
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import segmentation
from skimage.transform import resize
import time
from skimage.filters import gaussian 
import random

class Fluo_C3DH_A549_seg_train(Dataset):
    def __init__(self,data_path):
        #read segmentation file
        self.data_path = data_path
        self.data_names = []
        self.seg_names = []
        subdir = next(os.walk(self.data_path))[1]
        length = 0
        for name in subdir:
            if len(name)==5:
                dirname = os.path.join(self.data_path,name)
                seg_or_tra = next(os.walk(dirname))[1]
                for sub in seg_or_tra:
                    if sub == 'SEG':
                        dirname = os.path.join(dirname,sub)
                        _,_,data_files = next(os.walk(dirname))      
                        for each_data in data_files: 
                            data_fullname = os.path.join(dirname,each_data)
                            self.seg_names.append(data_fullname)

    def __len__(self):
        return len(self.seg_names)

    def __getitem__(self,inx):
        seg = sitk.ReadImage(self.seg_names[inx])
        seg = sitk.GetArrayFromImage(seg)
        #read image files
        _,filename =os.path.split(self.seg_names[inx])
        filename = os.path.splitext(filename)[0]
        subdir = next(os.walk(self.data_path))[1]
        for name in subdir:
            if len(name)==2:
                img_path = os.path.join(self.data_path,name)
                _,_,files = next(os.walk(img_path))
            for img_file in files:
                #print filename[7:len(filename)]
                #print img_file[1:len(img_file)-4]
                if filename[7:len(filename)] == img_file[1:len(img_file)-4]:
                    img = sitk.ReadImage(os.path.join(img_path,img_file))
                    img=sitk.GetArrayFromImage(img)
                    break
        z,x,y = seg.shape
        #img = img[start_z:start_z+16,:,:]
        img = img.astype('float32')/(2**16-1)
        seg = seg.astype('float32')
        sample = {}
        sample['data'] = img[None,...]
        sample['seg'] = seg[None,...]
        return sample

        
class cell_testing(Dataset):
    def __init__(self,data_path):
        self.data_path = data_path
        self.files = os.listdir(self.data_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self,inx):
        full_path = os.path.join(self.data_path,self.files[inx]) 
        img = sitk.ReadImage(full_path)
        img = sitk.GetArrayFromImage(img)
        img = img.astype('float32')
        img = img/(2**16-1)
        img = gaussian(img)
        #plt.imshow(img[0,:,:],cmap='gray')
        #plt.show()
        z,x,y = img.shape
        int_z = int(np.power(2,np.ceil(np.log2(z))))
        image_inter = resize(img,(5*z,512,512))
        expo = np.ceil(np.log2(5*z))
        z_dim = 2**expo
        z_dim = int(z_dim)
        image_new = np.zeros((z_dim,x,y))
        image_new[0:5*z] = image_inter
        #img = image_new.astype('float32')
        #plt.imshow(seg[0,:,:],cmap='gray')
        #plt.show()
        sample = {}
        sample['seg'] = []
        sample['data'] = image_new[None,...]
        sample['name'] = self.files[inx]
        sample['image_size'] = [z,x,y]
        return sample

class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')
        self.X = X
        self.slices, rows, cols= X.shape
        self.ind = self.slices//2
        self.im = ax.imshow(self.X[self.ind,:, :],cmap='gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()
    def update(self):
        self.im.set_data(self.X[self.ind,:, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

class IndexTracker2(object):
    def __init__(self, ax, X,Y):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')
        self.X = X
        self.slices, rows, cols= X.shape
        self.ind = self.slices//2
        self.im = ax.imshow(self.X[self.ind,:, :],cmap='gray')
        #self.cmap2 = cmap2
        self.Y = Y
        self.im2 = ax.imshow(self.Y[self.ind,:,:],cmap='gray',alpha=0.3)
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()
    def update(self):
        self.im.set_data(self.X[self.ind,:, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


class IndexTracker2plot(object):
    def __init__(self, ax1, ax2,X,Y):
        self.ax1 = ax1
	self.ax2 = ax2
        ax1.set_title('use scroll wheel to navigate images')
	ax2.set_title('use scroll wheel to navigate images')
        self.X = X
	self.Y = Y
        self.slices, rows, cols= X.shape
        self.ind = self.slices//2
        self.im1 = ax1.imshow(self.X[self.ind,:, :],cmap='gray')
	self.im2 = ax2.imshow(self.Y[self.ind,:, :],cmap='gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()
    def update(self):
        self.im1.set_data(self.X[self.ind,:, :])
	self.im2.set_data(self.Y[self.ind,:, :])
        self.ax1.set_ylabel('slice %s' % self.ind)
	self.ax2.set_ylabel('slice %s' % self.ind)
        self.im1.axes.figure.canvas.draw()
	self.im2.axes.figure.canvas.draw()

if __name__=='__main__':
    cell = Fluo_C3DH_A549_seg_train('/home/tom/Modified-3D-UNet-Pytorch/Fluo-C3DH-A549/')
    img1 = cell.__getitem__(0)['data']
    img = img1[0]
    img = img[0]
    plt.imshow(img)
    plt.show()
