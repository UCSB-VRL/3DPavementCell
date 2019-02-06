import numpy as np
from skimage.morphology import label
import SimpleITK as sitk
from celldataset import IndexTracker,IndexTracker2
from matplotlib import pyplot as plt
from skimage.filters import threshold_local
from skimage.transform import resize
from skimage.morphology import closing,binary_closing,binary_opening
from skimage.segmentation import morphological_chan_vese,chan_vese
from skimage import segmentation
from skimage.feature import peak_local_max
from scipy.signal import medfilt
from scipy import ndimage as ndi
from skimage.measure import label
from celldataset import IndexTracker
import cv2
from matplotlib import pyplot as plt
#import morphsnakes as ms
from region_grow import regionGrowing
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

    seg = sitk.ReadImage('/home/tom/celldata/backgroundSubtracted_T01.tif')
    seg = sitk.GetArrayFromImage(seg)
    seg = seg.astype('float')
    seg = seg/255
    x_start = int(6.54/20.16*512)
    x_end = int(17.56/20.16*512)
    y_start = int(6.10/20.16*512)
    y_end = int(15.19/20.16*512)
    '''
    shot = np.zeros((y_end-y_start,x_end-x_start))
    seg_sli = seg[0]
    #seg_sli[seg_sli==1]=0
    #seg_sli[seg_sli==18]=0
    shot = seg_sli[y_start:y_end,x_start:x_end]
    #shot[shot>0]=1
    #distance = ndi.distance_transform_edt(1-shot)
    #cells = peak_local_max(distance,indices=False)
    #distance[distance>0]=1
    #masks = label(distance)
    #masks = watershed(shot,masks,watershed_line=False)
    #line = segmentation.find_boundaries(shot,connectivity=1,mode='thick')
    #line = line*1
    #print np.unique(shot)
    fig = plt.imshow(shot,cmap='gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()
    exit(0)
    '''


    #above is for paper
    minimum = 50
    slice_ind = 0
    seg =  sitk.ReadImage('/home/tom/result/backgroundSubtracted_T01.timatched.tifseg.tif')
    img = sitk.ReadImage('/home/tom/result/backgroundSubtracted_T01.tifprob.tif')
    nm = 'cleaned_SB382_10DAG_0.15%Pect_P1C1F2c561.tif'
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
    img = resize(img,(img.shape[0]/5,img.shape[1],img.shape[2]))
    prob_map = seg_new
    seg_new_img = sitk.GetImageFromArray((seg_new*255).astype('uint8'))
    sitk.WriteImage(seg_new_img,'/home/tom/result/'+filename+'prob.tif')
    sitk.WriteImage(seg_new_img,'/home/tom/result/'+filename+'prob.nii')
    seg_old = seg_new
    seg_new[seg_new<0.04] = 0
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
    '''for i in uni:
        print i
        if i==6:
            mask_temp[masks==i]=i
        elif i==0:
            mask_temp[masks==i]=i
        else:
            mask_temp[masks==i]=1
    '''
    masks = mask_temp
    plt.imshow(masks)
    plt.show()
    #mask = watershed(seg_new[slice_ind],masks)
    #one_cell = np.zeros_like(seg_new)
    #one_cell[slice_ind] = mask
    #masks = propagate(seg_new,one_cell,slice_ind)
    mask_3d  = np.zeros_like(seg_new)
    mask_3d[slice_ind] = masks
    last = np.zeros((512,512))
    first = np.zeros((512,512))
    #last[400:402,259:262]=30
    first[274,243]=1
    #mask_3d[5]=last
    #mask_3d[18*5]=last
    mask_3d[0,372:375,242:244]=30
    #masks_img = sitk.GetImageFromArray((masks*255).astype('uint8'))
    #sitk.WriteImage(masks_img,'/home/tom/result/'+'seeds.png')
    #masks = random_walker(seg_new,mask_3d,beta=10,mode='bf')
    print np.unique(mask_3d)
    masks = watershed(seg_new,mask_3d,watershed_line=True)
    print np.unique(masks)
    #one_cell = (masks==11)*1
    masks[masks==30]=0
    '''#CRF
    #distance_one_cell = ndi.distance_transform_edt(one_cell)
    #distance_one_cell[distance_one_cell>15] = np.amax(distance_one_cell)
    #distance_one_cell = distance_one_cell/np.amax(distance_one_cell)
    prob_map = np.multiply(1-prob_map,one_cell)
    pro = CRFProcessor.CRF3DProcessor()
    seg_new = np.transpose(prob_map,(1,2,0))
    labels = np.zeros((512,512,12,2))
    seg_new[seg_new>0.01] = 1
    labels[:,:,:,0] = seg_new
    labels[:,:,:,1] = 1-seg_new
    #distance_one_cell = np.transpose(distance_one_cell,(1,2,0))
    result = pro.set_data_and_run(seg_new,labels)
    result = np.transpose(result,(2,0,1))
    print np.unique(result)
    #plt.imshow(mask_mid_bin)
    #plt.show()
    #masks = resize(masks,(masks.shape[0]/5,masks.shape[1],masks.shape[2]),mode='constant')
    masks_img = sitk.GetImageFromArray((masks).astype('uint8'))
    sitk.WriteImage(masks_img,'/home/tom/1_8_result/'+filename+'seg.tif')'''
    fig = plt.imshow(masks[0,y_start:y_end,x_start:x_end])
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    #print np.unique(masks)
    fig,ax = plt.subplots(1,1)
    tracker = IndexTracker(ax,masks)
    fig.canvas.mpl_connect('scroll_event',tracker.onscroll)
    plt.show()
    exit(0)
    #seg_new[seg_new>1]=1
    plt.imshow(max_proj,cmap='gray')
    seg_new_down = resize(seg_new,(seg_new.shape[0]/5,seg_new.shape[1],seg_new.shape[2]))
    seg_new_img = sitk.GetImageFromArray((seg_new_down*255).astype('uint8'))
    sitk.WriteImage(seg_new_img,'/home/tom/1_8_result/'+filename+'prob.tif')
    markers = np.zeros(seg_new.shape)
    z1,y1,x1 = 50,316,178
    z,y,x = np.indices(seg_new.shape)
    circle  = ((z-z1)**2+(y-y1)**2+(x-x1)**2)<5**2
    #circle2 = ((z-z2)**2+(y-y2)**2+(x-x2)**2)<5**2
    markers = np.logical_or(circle,markers)
    #markers = np.logical_or(circle2,markers)
    markers = markers*1
    #markers = label(markers)
    markers = (markers).astype(float)
    #segmentation = morphological_chan_vese(seg_new,iterations = 100,init_level_set=markers)
    fig,ax = plt.subplots(1,1)
    tracker = IndexTracker(ax,segmentation)
    fig.canvas.mpl_connect('scroll_event',tracker.onscroll)
    plt.show()
    exit(0)

    #plt.imshow(markers[50],cmap='gray')
    #plt.show()
    #print seg_new.shape
    #one_cell = regionGrowing(seg_new_down,[z1/5,y1,x1],0)
    #print np.unique(one_cell)
    #plt.imshow(one_cell[10],cmap='gray')
    #plt.show()
    #print one_cell.shape
    #average_proj = np.sum(seg_new,0)
    #local_maxi = peak_local_max(-average_proj, indices=False, footprint=np.ones((3, 3)), labels=average_proj)
    #print len(local_maxi)
    #plt.imshow(average_proj,cmap='gray')
    #plt.show()
    
    #one_cell = grow(seg_new,(50,312,189),3)
    mask = np.zeros_like(seg_new,np.uint8)
    mask[50][312][189]=1
    one_cell = geodesic_distance.geodesic3d_raster_scan(seg_new,mask,0.4,5)
    #init_ls = ms.circle_level_set(seg_new.shape, (50,312,189), 10)
    #callback = visual_callback_3d(plot_each=20)
    #one_cell = ms.morphological_geodesic_active_contour(-seg_new, iterations=30,init_level_set=init_ls,smoothing=1)
    #one_cell = 1*one_cell
    #one_cell = (init_ls*255).astype('uint8')
    fig,ax = plt.subplots(1,1)
    tracker = IndexTracker(ax,one_cell)
    fig.canvas.mpl_connect('scroll_event',tracker.onscroll)
    plt.show()
    exit(0)
    one_cell = sitk.GetImageFromArray(one_cell)
    sitk.WriteImage(one_cell,'/home/tom/result/'+'one_cell.tif')
    exit(0)



    seg_new = seg_new.astype('uint8')
    seg_new_img = sitk.GetImageFromArray(seg_new)
    sitk.WriteImage(seg_new_img,'/home/tom/result/'+'overlay.tif')
    seg = np.zeros(seg_new.shape)
    for i in range(len(seg)):
        img_sli = seg_new[i]
        laplacian = cv2.Laplacian(img_sli,cv2.CV_64F)
        laplacian = laplacian/np.amax(laplacian)
        seg[i] = laplacian
        #plt.imshow(np.abs(laplacian),cmap='gray')
        #plt.show()

        #im_sli,contours,hierarchy = cv2.findContours(img_sli,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #seg[i] = img_sli
        #print (len(contours))
        #cv2.drawContours(img_sli,contours,-1, (0,255,0), 3)
        #cv2.imshow("contours",img_sli)
        #cv2.waitKey(0)

    #print np.amax(seg)
    seg_new = np.abs(seg)*255
    seg = (seg_new).astype('uint8')
    seg_img = sitk.GetImageFromArray(seg)
    #seg = sitk.Cast(seg,sitk.Float32)
    sitk.WriteImage(seg_img,'/home/tom/result/'+'contour.tif')
    #new_seg_img = sitk.GetImageFromArray(seg_new)
    #sitk.WriteImage(new_seg_img,'/home/tom/result/'+'segmentation.tif')
    #exit(0)
    #print np.amax(seg_new)
    threshold = 5
    #threshold = threshold_local(seg_new, block_size=35, offset=10)
    dist  = np.zeros(seg_new.shape)
    for i in range(len(seg)):
        img_sli = seg_new[i]
        img_thresh = threshold_local(img_sli,block_size=11,offset=1)
        img_sli_thresh = np.zeros(img_sli.shape)
        img_sli_thresh = img_sli>threshold
        img_sli_dist =  ndi.distance_transform_edt(1-img_sli_thresh)
        #plt.imshow(img_sli_dist,cmap='gray')
        #plt.show()
        dist[i] = img_sli_dist
    #seg = seg_new>threshold
    #seg = medfilt(seg,3)
    dist = resize(dist,(dist.shape[0]/5,dist.shape[1],dist.shape[2]))
    #for i in range(len(dist)):

    #print np.amax(seg)
    #seg = 1-seg
    #labels = label(seg,background=0)
    '''#print np.amax(labeled)
    fig,ax = plt.subplots(1,1)
    tracker = IndexTracker(ax,labels)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()'''
main()
