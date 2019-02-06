import numpy as np
import SimpleITK as sitk
import random
from skimage import segmentation
from skimage.transform import resize
import os
from celldataset import IndexTracker2plot
from matplotlib import pyplot as plt

# Given the input of np 3D image array, return certain number of image patches in a list  
def random_crop(image_stack,seg_stack,num_patch):
    imgpat = []
    segpat = []
    z,y,x = image_stack.shape
    #crop center 
    for i in range(num_patch):
        #random size
        cut_z = random.randint(64,512)
        cut_y = random.randint(64,512)
        cut_x = random.randint(64,512)
        rangez = (z - cut_z) // 2 if z>cut_z else 0
        rangey = (y - cut_y) // 2 if y>cut_y else 0
        rangex = (x - cut_x) // 2 if x>cut_x else 0
        offsetz = 0 if rangez == 0 else np.random.randint(0,rangez)
        offsety = 0 if rangey == 0 else np.random.randint(0,rangey)
        offsetx = 0 if rangex == 0 else np.random.randint(0,rangex)
        cropped_image = image_stack[offsetz:offsetz+cut_z, offsety:offsety+cut_y, offsetx:offsetx+cut_x]
        cropped_seg = seg_stack[offsetz:offsetz+cut_z, offsety:offsety+cut_y, offsetx:offsetx+cut_x]
        cropped_image = resize(cropped_image,(32,32,32),order=0)
        imgpat.append(cropped_image)
        cropped_seg = resize(cropped_seg,(32,32,32),order=0)
        segpat.append(cropped_seg)
    return imgpat,segpat

#crop PNAS dataset
def crop_PNAS(path):
    img_patches = []
    seg_patches = []
    subdir = next(os.walk(path))[1]
    #print(sorted(subdir))
    for name in subdir:
        dirname = os.path.join(path,name)
        data_file = "processed_tiffs"
        data_file = os.path.join(dirname,data_file)
        _,_,data_files = next(os.walk(data_file))                                               
        for each_data in data_files:    
            if "acylYFP" in each_data:  
                data_fullname = os.path.join(data_file,each_data)
                each_data_without = os.path.splitext(each_data)[0]
                img = sitk.ReadImage(data_fullname)
                img = sitk.GetArrayFromImage(img)
                _,filename =os.path.split(data_fullname)
                filename = os.path.splitext(filename)[0]
                plant = filename.split('_')[1]
                seg_path = os.path.join(path,plant)
                seg_path = os.path.join(seg_path,"segmentation_tiffs")
                _,_,files = next(os.walk(seg_path))
                for seg_file in files:
                    if each_data_without == seg_file[0:len(each_data_without)]:
                        seg = sitk.ReadImage(os.path.join(seg_path,seg_file))       
                        seg =  sitk.GetArrayFromImage(seg)
                        seg = segmentation.find_boundaries(seg,connectivity=1,mode='thick')
                        break
                imgpat,segpat = random_crop(img,seg,500)
                img_patches.append(imgpat)
                seg_patches.append(segpat)
    img_patches = np.concatenate(img_patches)
    seg_patches = np.concatenate(seg_patches)
    np.save('/home/tom/Modified-3D-UNet-Pytorch/transformation/inputs.npy',img_patches)
    np.save('/home/tom/Modified-3D-UNet-Pytorch/transformation/ground_truth.npy',seg_patches)
def main():
    #crop_PNAS('/home/tom/Modified-3D-UNet-Pytorch/PNAS')
    img_patches = np.load('/home/tom/Modified-3D-UNet-Pytorch/transformation/inputs.npy')
    seg_patches = np.load('/home/tom/Modified-3D-UNet-Pytorch/transformation/ground_truth.npy')
    #print len(img_patches)
    #print len(seg_patches)
    img = img_patches[500]
    seg = seg_patches[500]
    fig,(ax1,ax2) = plt.subplots(2,1)
    tracker = IndexTracker2plot(ax1,ax2,img,seg)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

main()

