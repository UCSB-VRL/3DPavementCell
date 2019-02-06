import numpy as np
from skimage.morphology import label
import SimpleITK as sitk
from celldataset import IndexTracker,IndexTracker2
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
#from matplotlib import pyplot as plt
#import morphsnakes as ms
#from region_grow import regionGrowing
from skimage.feature import peak_local_max
from skimage.segmentation import watershed,random_walker,find_boundaries
#def visual_callback_3d(fig=None, plot_each=1):
#import geodesic_distance
import os
#from denseinference import CRFProcessor


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
    #some hyperparameters
    minimum = 50
    slice_ind = 6
    directory = 'prob_map'
    blk_threshold = 0.05
    min_dis = 10
    label_thresh = 10

    #files = os.listdir(directory)
    img = sitk.ReadImage('/home/tom/celldata/data4_sub/T01sub.tif')
    img = sitk.GetArrayFromImage(img)
    img = img.astype('float32')
    img_img = img/np.amax(img)
    img_img = sitk.GetImageFromArray(img_img)
    sitk.WriteImage(img_img,'/home/tom/celldata/data4_sub/'+'T01sub.nii') 
    img[img<2500]=0
    #img[img>6000]=1
    seg_new = img/np.amax(img)
    #seg_new[seg_new<0.]=0
    mask_3d  = np.zeros_like(seg_new)
    fig,ax = plt.subplots(1,1)
    tracker = IndexTracker(ax,seg_new)
    fig.canvas.mpl_connect('scroll_event',tracker.onscroll)
    plt.show()
    points = [[27,52],[132,53],[297,39],[364,31],[460,31],[282,123],[351,128],[419,162],[488,185],[132,238],[275,291],[385,287],[35,350],[187,457],[428,448]]
    r = 20
    markers = mask_3d[0]
    for k,(x,y) in enumerate(points):
        cv2.circle(markers,(x,y),r,k+1,-1)
    markers = markers.astype(int)
    mask_3d[0] = markers
    mask_3d = watershed(seg_new,mask_3d)
    fig,ax = plt.subplots(1,1)
    tracker = IndexTracker(ax,mask_3d)
    fig.canvas.mpl_connect('scroll_event',tracker.onscroll)
    #plt.imshow(markers,cmap='gray')
    plt.show()
    #mask_3d[] = 
    exit(0)
    for file in files:
        print file
        slice_ind = input('Please input the slice number for seeds:')
        blk_threshold = input('Please input the threhold for black voxels')
        slice_ind = slice_ind-1
        path  = os.path.join(directory,file)
        img = sitk.ReadImage(path)
        img = sitk.GetArrayFromImage(img)
        img = img.astype('float32')
        seg_new = img/np.amax(img)
        seg_new[seg_new<blk_threshold] = 0
        mid_slice = seg_new[slice_ind]
        mask_mid = peak_local_max(-mid_slice,min_distance = min_dis,indices=False,exclude_border=1)
        #plt.imshow(mid_slice,cmap='gray')
        #plt.show()
        mask_mid = mask_mid*1
        mask_mid  = binary_opening(1-mask_mid)
        mask_mid = binary_closing(mask_mid) 
        distance = ndi.distance_transform_edt(1-mask_mid) 
        #mask_mid = peak_local_max(distance,min_distance=30,indices=False,threshold_rel=0.2)
        mask_mid_bin = np.zeros_like(distance)
        mask_mid_bin[distance>label_thresh]=1
        masks = label(mask_mid_bin)
        #masks_img = sitk.GetImageFromArray((masks*255).astype('uint8'))
        #sitk.WriteImage(masks_img,'/home/tom/result/'+filename+'det.png')
        uni,counts = np.unique(masks,return_counts=True)
        for i in uni:
            if counts[i]<minimum:
                masks[masks==i]=0
        uni,counts = np.unique(masks,return_counts=True)
        mask_temp = masks
        masks = mask_temp
        #plt.imshow(masks)
        #plt.show()
        #mask = watershed(seg_new[slice_ind],masks)
        #one_cell = np.zeros_like(seg_new)
        #one_cell[slice_ind] = mask
        #masks = propagate(seg_new,one_cell,slice_ind)
        mask_3d  = np.zeros_like(seg_new)
        mask_3d[slice_ind] = masks
        #last = np.zeros((512,512))
        #first = np.zeros((512,512))
        #last[400:402,259:262]=30
        #first[375:380,245:250]=1
        #mask_3d[5]=last
        #mask_3d[18*5]=last
        #mask_3d[0]=first
        #masks_img = sitk.GetImageFromArray((masks*255).astype('uint8'))
        #sitk.WriteImage(masks_img,'/home/tom/result/'+'seeds.png')
        #masks = random_walker(seg_new,mask_3d,beta=10,mode='bf')
        masks = watershed(seg_new,mask_3d,watershed_line=True)
        #CRF
        #distance_one_cell = ndi.distance_transform_edt(one_cell)
        #distance_one_cell[distance_one_cell>15] = np.amax(distance_one_cell)
        #distance_one_cell = distance_one_cell/np.amax(distance_one_cell)
        #prob_map = np.multiply(1-prob_map,one_cell)
        #pro = CRFProcessor.CRF3DProcessor()
        #seg_new = np.transpose(prob_map,(1,2,0))
        #labels = np.zeros((512,512,12,2))
        #seg_new[seg_new>0.01] = 1
        #labels[:,:,:,0] = seg_new
        #labels[:,:,:,1] = 1-seg_new
        #distance_one_cell = np.transpose(distance_one_cell,(1,2,0))
        #result = pro.set_data_and_run(seg_new,labels)
        #result = np.transpose(result,(2,0,1))
        #print np.unique(result)
        #plt.imshow(mask_mid_bin)
        #plt.show()
        #masks = resize(masks,(masks.shape[0]/5,masks.shape[1],masks.shape[2]),mode='constant')
        masks_img = sitk.GetImageFromArray((masks).astype('uint8'))
        sitk.WriteImage(masks_img,'result/'+file+'seg.nii')

main()
