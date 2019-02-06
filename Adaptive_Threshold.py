import numpy as np
from skimage.morphology import label
import SimpleITK as sitk
from celldataset import IndexTracker
from matplotlib import pyplot as plt
from skimage.filters import threshold_local
from skimage.transform import resize
from skimage.morphology import closing,binary_closing,binary_opening
from skimage.segmentation import morphological_chan_vese,chan_vese
from skimage.feature import peak_local_max
from scipy.signal import medfilt
from scipy import ndimage as ndi
from skimage.measure import label
from celldataset import IndexTracker
import cv2
from matplotlib import pyplot as plt
import morphsnakes as ms
from region_grow import regionGrowing
from skimage.feature import peak_local_max
from skimage.segmentation import watershed,random_walker,find_boundaries
#def visual_callback_3d(fig=None, plot_each=1):
import geodesic_distance
import os

def maxprojection(img):
    z,x,y = img.shape
    max_proj = np.zeros((x,y))
    max_proj = np.amax(img,axis=0)
    max_proj = max_proj.astype('float32')
    return max_proj

#def gvf(img):

def propagate(img,one_cell,sli_ind):
    z,x,y = img.shape
    sli_start,sli_end = sli_ind,sli_ind
    boundaries = find_boundaries(one_cell[sli_ind])
    #print boundaries*1
    #plt.imshow(boundaries,cmap='gray')
    #plt.show()
    one_cell[sli_ind]=boundaries*1
    while (sli_start>0 or sli_end<z-1):
        print sli_start
        if sli_start>0:
            cur = img[sli_start]
            cur_2 = one_cell[sli_start]
            points = np.transpose(np.nonzero(cur_2))
            prev_sli = np.zeros((x,y))
            for point in points:
                #print len(point)
                print point
                prev = img[sli_start-1]
                #print cur[point].shape
                diff = abs(prev[max(point[0]-2,0):min(point[0]+3,x),max(point[1]-2,0):min(point[1]+3,y)]-cur[point[0],point[1]])
                #print diff.shape
                ind = np.unravel_index(np.argmin(diff, axis=None), diff.shape)
                prev_sli[ind[0]+point[0]-2,ind[1]+point[1]-2]=1
                one_cell[sli_start-1]=prev_sli
            sli_start = sli_start-1
        if sli_end<z-1:
            cur = img[sli_end]
            cur_2 = one_cell[sli_start]
            points = np.transpose(np.nonzero(cur_2))
            prev_sli = np.zeros((x,y))
            for point in points:
                prev = img[sli_start-1]
                diff = abs(prev[max(point[0]-2,0):min(point[0]+3,x),max(point[1]-2,0):min(point[1]+3,y)]-cur[point[0],point[1]])
                ind = np.unravel_index(np.argmin(diff, axis=None), diff.shape)
                prev_sli[ind[0]+point[0]-4,ind[1]+point[1]-4]=1
                one_cell[sli_end+1]=prev_sli
            sli_end = sli_end+1
    return one_cell





def main():
    minimum = 50
    slice_ind = 9
    img = sitk.ReadImage('/home/tom/result/backgroundSubtracted_T01.timatched.tif')
    seg =  sitk.ReadImage('/home/tom/result/backgroundSubtracted_T01.timatched.tifseg.tif')
    nm = '/home/tom/celldata/cleaned_SB382_10DAG_0.15%Pect_P3C2F6c561.tif'
    #original = sitk.ReadImage(nm)
    _,filename = os.path.split(nm)
    #sitk.WriteImage(original,'/home/tom/celldata/ori1.nii')
    #exit(0)
    img = sitk.GetArrayFromImage(img)
    img = img.astype('float32')
    seg = sitk.GetArrayFromImage(seg)
    seg = seg.astype('float32')
    seg = seg/255.0
    #seg = np.ones(seg.shape)
    seg_new = np.multiply(img,seg)
    #seg_new = np.multiply(seg_new,img)
    #seg_new = np.multiply(seg_new,img)
    seg_new = seg_new/255
    print seg_new.shape
    seg_new = resize(seg_new,(seg_new.shape[0]/5,seg_new.shape[1],seg_new.shape[2]))
    seg_new_img = sitk.GetImageFromArray((seg_new*255).astype('uint8'))
    sitk.WriteImage(seg_new_img,'/home/tom/result/'+filename+'prob.tif')
    seg_old = seg_new
    seg_new[seg_new<0.08] = 0
    fig,ax = plt.subplots(1,1)
    tracker = IndexTracker(ax,seg_new)
    fig.canvas.mpl_connect('scroll_event',tracker.onscroll)
    plt.show()
    exit(0)

    mid_slice = seg_new[slice_ind]
    mask_mid = peak_local_max(-mid_slice,min_distance = 10,indices=False,exclude_border=1)
    #plt.imshow(mid_slice,cmap='gray')
    #plt.show()
    mask_mid = mask_mid*1
    mask_mid  = binary_opening(1-mask_mid)
    mask_mid = binary_closing(mask_mid) 
    distance = ndi.distance_transform_edt(1-mask_mid) 
    #mask_mid = peak_local_max(distance,min_distance=30,indices=False,threshold_rel=0.2)
    mask_mid_bin = np.zeros_like(distance)
    mask_mid_bin[distance>10]=1
    masks = label(mask_mid_bin)
    masks_img = sitk.GetImageFromArray((masks*255).astype('uint8'))
    sitk.WriteImage(masks_img,'/home/tom/result/'+filename+'det.png')
    uni,counts = np.unique(masks,return_counts=True)
    for i in uni:
        if counts[i]<minimum:
            masks[masks==i]=0
    uni,counts = np.unique(masks,return_counts=True)
    mask_temp = masks
    for i in uni:
        print i
        if i==6:
            mask_temp[masks==i]=i
        elif i==0:
            mask_temp[masks==i]=i
        else:
            mask_temp[masks==i]=1
    masks = mask_temp
    plt.imshow(masks)
    plt.show()
    mask = watershed(seg_new[slice_ind],masks)
    one_cell = np.zeros_like(seg_new)
    one_cell[slice_ind] = mask
    masks = propagate(seg_new,one_cell,slice_ind)

main()
